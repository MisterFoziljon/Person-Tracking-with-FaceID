import os
import cv2
import time
import torch
import onnxruntime
import numpy as np
from scrfd import SCRFD
from ultralytics import YOLO
from collections import defaultdict
from arcface_onnx import ArcFaceONNX
from tracker.botsort import BoTSORT

from shapely.geometry.polygon import Polygon

from vision import Point
from vision import LineCounter
from vision import BoxAnnotator
from vision import ColorPalette
from vision import LineCounterAnnotator

onnxruntime.set_default_logger_severity(3)

class INOUT:
    def __init__(self):
        self.DETECTOR = SCRFD("/home/foziljon/PROJECTS/ATTENDENCE_SYSTEM/models/det_500m.onnx")
        self.DETECTOR.prepare(-1)
        self.RECOGNITION = ArcFaceONNX("/home/foziljon/PROJECTS/ATTENDENCE_SYSTEM/models/w600k_mbf.onnx")
        self.RECOGNITION.prepare(-1)
        
        self.MODEL = YOLO("/home/foziljon/PROJECTS/ATTENDENCE_SYSTEM/models/yolov8n.pt")
        self.MODEL.fuse()
        
        self.DATABASE = self.database_loader('/home/foziljon/PROJECTS/ATTENDENCE_SYSTEM/dataset')
        self.BOX_ANNOTATOR = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
        self.LINE_ANNOTATOR = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.IN_LINE_COUNTER = LineCounter(start=Point(910,700), end=Point(2240,700))
        self.OUT_LINE_COUNTER = LineCounter(start=Point(2880,890), end=Point(2340,1630))
        
        self.POLYGON = Polygon([(770,890),(1610,530),(3270,1010),(2790,1860)])
        
        self.OUT_RECTANGLE = np.array([[[770,890],[1610,530],[3270,1010],[2790,1860]]])
        self.IN_RECTANGLE = np.array([[[910,0],[2240,0],[2240,1400],[910,1400]]])

    def database_loader(self, database_folder):
        database = {}

        for person_name in os.listdir(database_folder):
            person_path = os.path.join(database_folder, person_name)
            
            if os.path.isdir(person_path):
                for image_file in os.listdir(person_path):
                    if image_file.startswith("."):
                        continue

                    image_path = os.path.join(person_path, image_file)
                    img = cv2.imread(image_path) 

                    bboxes, kpss = self.DETECTOR.autodetect(img, 1)

                    if bboxes.shape[0]==0:
                        return -1.0, "Face not found in Image"

                    feat = self.RECOGNITION.get(img, kpss[0])
                    database[person_name] =  feat
        return database
    
    def get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

    def IN(self, source):
        cap = cv2.VideoCapture(source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        FACEID = defaultdict(str)
        fps = cap.get(cv2.CAP_PROP_FPS)

        TRACKER = BoTSORT(frame_rate=fps)
        xpmin,ypmin,xpmax,ypmax = self.IN_RECTANGLE[0][0][0],self.IN_RECTANGLE[0][0][1],self.IN_RECTANGLE[0][2][0],self.IN_RECTANGLE[0][2][1]
        count, FPSs= 0,0
        while cap.isOpened():
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            count+=1
            polygon = frame[ypmin:ypmax,xpmin:xpmax]
            results = self.MODEL.predict(polygon, conf=0.5, stream = False, device = torch.device("cpu"), verbose=False, classes=[0])

            xyxy=results[0].boxes.xyxy.cpu().numpy()
            xyxy += np.array([xpmin,ypmin,xpmin,ypmin])

            confidence=results[0].boxes.conf.cpu().numpy()
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)

            online_targets = TRACKER.update(xyxy, confidence, class_id, frame)

            img = np.ascontiguousarray(np.copy(frame))

            BBOX = []
            TRACKER_ID = []
            COLOR = []
            for parameters in online_targets:

                TRACKER_ID.append(int(parameters.track_id))
                COLOR.append(self.get_color(int(parameters.track_id)))
                             
                x1, y1, w, h = parameters.tlwh
                xmin,ymin,xmax,ymax = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                xmin = max(0,xmin)
                ymin = max(0,ymin)
                xmax = min(xmax,width)
                ymax = min(ymax,height)

                BBOX.append([xmin,ymin,xmax,ymax])

            for bbox, tracker_id in zip(BBOX,TRACKER_ID):
                if len(bbox)==0 or FACEID[tracker_id]!="":
                    continue
                
                xmin,ymin,xmax,ymax = bbox
                person = frame[ymin:ymax,xmin:xmax]

                faces_coords, kpss = self.DETECTOR.autodetect(person, max_num=10)

                if len(faces_coords)==0:
                    continue

                first = True
                check = 0
                ID = "Unknown"
                
                for face_coords,kps in zip(faces_coords,kpss):

                    feat = self.RECOGNITION.get(person, kps)
                    maxi = 0

                    for identity, db_feat in self.DATABASE.items():
                        similarity = self.RECOGNITION.compute_sim(feat, db_feat)
                        if similarity > maxi:
                            maxi = similarity
                            ids = identity

                    if first:
                        check = face_coords[1]
                        ID = ids
                        first = False

                    if check>face_coords[1]:
                        check = face_coords[1]
                        ID = ids

                if maxi < 0.4:
                    FACEID[tracker_id] = ""
                else:
                    FACEID[tracker_id] = ID
                
            faceIDs = np.array([FACEID[ids] if FACEID[ids]!='' else 'Unknown' for ids in TRACKER_ID])
            labels = [f"#{trackerID} {faceID}" for faceID, trackerID  in zip(faceIDs,TRACKER_ID)]

            self.IN_LINE_COUNTER.update(BBOX, TRACKER_ID, face_id=FACEID, out = False)
            self.LINE_ANNOTATOR.annotate(frame=frame, line_counter=self.IN_LINE_COUNTER, text = "IN")

            frame = self.BOX_ANNOTATOR.annotate(frame=frame, bboxes=BBOX, colors = COLOR, labels=labels)
            cv2.polylines(frame, [self.IN_RECTANGLE], True, (255,255,255), thickness=5)
            end = time.time()
            cv2.putText(frame, f"FPS: {1./(end-start)}", (60,60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)
            FPSs+=1./(end-start)
            frame = cv2.resize(frame,(width//3,height//3),interpolation = cv2.INTER_AREA)
            cv2.imshow("video",frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        print(FPSs/count)
if __name__=="__main__":
    action = INOUT()
    action.IN("sources/in.mp4")
