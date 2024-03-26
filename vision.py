import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Tuple

DEFAULT_COLOR_PALETTE = ["#e6194b","#3cb44b","#ffe119","#0082c8","#f58231","#911eb4","#46f0f0","#f032e6","#d2f53c","#fabebe","#008080","#e6beff","#aa6e28","#fffac8","#800000","#aaffc3"]

@dataclass
class Color:
    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, color_hex: str):
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)

@dataclass
class ColorPalette:
    colors: List[Color] = field(default_factory=lambda: [Color.from_hex(color_hex) for color_hex in DEFAULT_COLOR_PALETTE])

    @classmethod
    def from_hex(cls, color_hex_list: List[str]):
        colors = [Color.from_hex(color_hex) for color_hex in color_hex_list]
        return cls(colors)

    def by_idx(self, idx: int) -> Color:
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        idx = idx % len(self.colors)
        return self.colors[idx]

@dataclass
class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class Vector:
    start: Point
    end: Point

    def is_in(self, point: Point) -> bool:
        v1 = Vector(self.start, self.end)
        v2 = Vector(self.start, point)
        cross_product = (v1.end.x - v1.start.x) * (v2.end.y - v2.start.y) - (v1.end.y - v1.start.y) * (v2.end.x - v2.start.x)
        return cross_product < 0


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    def pad(self, padding):
        return Rect(x=self.x - padding, y=self.y - padding, width=self.width + 2 * padding, height=self.height + 2 * padding)


class LineCounter:
    def __init__(self, start: Point, end: Point):
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0

    def update(self, boxes, trackerID , face_id, out):
        for xyxy, tracker_id in zip(boxes, trackerID):
            if tracker_id is None:
                continue
            x1, y1, x2, y2 = xyxy
            xmin = xmax = (x1+x2)/2.
            ymin = ymax = (y1+y2)/2.
            anchors = [Point(x=xmin-50, y=ymin-50), Point(x=xmin-50, y=ymax+50), Point(x=xmax+50, y=ymin-50), Point(x=xmax+50, y=ymax+50)]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            if len(set(triggers)) == 2:
                continue
                
            tracker_state = triggers[0]
            
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
                
            
            if not tracker_state and not out:
                self.out_count += 1
                #print(f'Saqlash   {face_id[tracker_id]}') 

            elif tracker_state and out:
                self.out_count += 1
                #print(f'Saqlash   {face_id[tracker_id]}') 


class LineCounterAnnotator:
    def __init__(self, thickness: float = 2, color: Color = (255,255,255), text_thickness: float = 2, text_color: Color = (0,0,0), text_scale: float = 0.5, text_offset: float = 1.5, text_padding: int = 10):

        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, line_counter: LineCounter, text: str) -> np.ndarray:
        cv2.line(frame, line_counter.vector.start.as_xy_int_tuple(), line_counter.vector.end.as_xy_int_tuple(), self.text_color, self.thickness, lineType=cv2.LINE_AA, shift=0)
        cv2.circle(frame, line_counter.vector.start.as_xy_int_tuple(), radius=5, color=self.text_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, line_counter.vector.end.as_xy_int_tuple(), radius=5, color=self.text_color, thickness=-1, lineType=cv2.LINE_AA)

        out_text = f"{text}: {line_counter.out_count}"

        (out_text_width, out_text_height), _ = cv2.getTextSize(out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)

        out_text_x = int((line_counter.vector.end.x + line_counter.vector.start.x - out_text_width) / 2)
        out_text_y = int((line_counter.vector.end.y + line_counter.vector.start.y + out_text_height) / 2 + self.text_offset * out_text_height)

        out_text_background_rect = Rect(x=out_text_x, y=out_text_y - out_text_height, width=out_text_width, height=out_text_height).pad(padding=self.text_padding)

        cv2.rectangle(frame, out_text_background_rect.top_left.as_xy_int_tuple(), out_text_background_rect.bottom_right.as_xy_int_tuple(), self.color, -1)

        cv2.putText(frame, out_text, (out_text_x, out_text_y), cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_color, self.text_thickness, cv2.LINE_AA)
        
        
class BoxAnnotator:
    def __init__(self, color: Union[Color, ColorPalette], thickness: int = 2, text_color: Color = (0,0,0), text_scale: float = 0.5, text_thickness: int = 1, text_padding: int = 10):

        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, bboxes: np.array, colors, labels: Optional[List[str]] = None) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for xyxy, label, color in zip(bboxes, labels, colors):
         
            x1, y1, x2, y2 = xyxy
            text_width, text_height = cv2.getTextSize(text=label, fontFace=font, fontScale=self.text_scale, thickness=self.text_thickness)[0]

            text_x = x1 + self.text_padding
            text_y = y1 - self.text_padding

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * self.text_padding - text_height

            text_background_x2 = x1 + 2 * self.text_padding + text_width
            text_background_y2 = y1

            cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=self.thickness)
            cv2.rectangle(img=frame, pt1=(text_background_x1, text_background_y1), pt2=(text_background_x2, text_background_y2), color=color, thickness=cv2.FILLED)
            cv2.putText(img=frame, text=label, org=(text_x, text_y), fontFace=font, fontScale=self.text_scale, color=(255,255,255), thickness=self.text_thickness, lineType=cv2.LINE_AA)
        return frame