import argparse
from vision import Point 
from shapely.geometry.polygon import Polygon
import os

def parse_args():
    PATH = os.getcwd()
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument(
        "--face_detector", 
        type=str, 
        default=os.path.join(PATH, 'models/det_500m.onnx'), 
        help="The path of the facial detection model")
    
    parser.add_argument(
        "--face_recognizer",
        type=str, 
        default=os.path.join(PATH, 'models/w600k_mbf.onnx'),
        help="The path of the facial recognition model")

    parser.add_argument(
        "--person_detector",
        type=str, 
        default=os.path.join(PATH, 'models/yolov8n.pt'),
        help="The path of the facial recognition model")
    
    parser.add_argument(
        "--in_rtsp",
        type=str,
        default=os.path.join(PATH, 'sources/in.mp4'),
        help="video source path or rtsp protocol")

    parser.add_argument(
        "--out_rtsp",
        type=str,
        default=os.path.join(PATH, 'sources/out.mp4'),
        help="video source path or rtsp protocol")
    
    parser.add_argument(
        "--use_cpu",
        type=bool, 
        default=True)
    
    parser.add_argument(
        "--database_folder", 
        type=str, 
        default=os.path.join(PATH, "dataset"))
    
    parser.add_argument(
        "--in_line_xy_min", 
        type=Point,
        default=Point(910,700))

    parser.add_argument(
        "--in_line_xy_max", 
        type=Point,
        default=Point(2240,700))
    
    parser.add_argument(
        "--out_line_xy_min", 
        type=Point,
        default=Point(2880,890))

    parser.add_argument(
        "--out_line_xy_max", 
        type=Point,
        default=Point(2340,1630))

    parser.add_argument(
        "--polygon", 
        type=Polygon,
        default=Polygon([(770,890),(1610,530),(3270,1010),(2790,1860)]))

    parser.add_argument(
        "--face_threshold",
        type=float,
        default=0.4)

    parser.add_argument(
        "--person_threshold",
        type=float,
        default=0.5)

    args = parser.parse_args()

    return args

def main():
    FLAGS = parse_args()

if __name__ == '__main__':
    main()
