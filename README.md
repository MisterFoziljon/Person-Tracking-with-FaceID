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
python IN.py
# python OUT.py 
```

### Parallel Usage:
```python
import threading
import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name])

if __name__ == "__main__":
    script1_thread = threading.Thread(target=run_script, args=("IN.py",))
    script2_thread = threading.Thread(target=run_script, args=("OUT.py",))

    script1_thread.start()
    script2_thread.start()

    script1_thread.join()
    script2_thread.join()
```
