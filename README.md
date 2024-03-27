# Person-Tracking-with-FaceID

### Requirements:
``` terminal
conda create --name env python=3.9
conda activate env
```

``` terminal
pip install opencv-python
pip install torchvision==0.14.1
pip install onnxruntime
pip install onnx
pip install ultralytics
pip install scikit-image
pip install lap
pip install cython_bbox
pip install yacs
pip install termcolor
pip install scikit-learn
pip install tabulate
pip install tensorboard
pip install shapely
pip install faiss-cpu
pip install numpy==1.23.1
```

### Usage:
``` python
from INOUT import INOUT

action = INOUT()

action.IN("sources/in.mp4") # in checker
# action.OUT("sources/out.mp4") # out checker
```
