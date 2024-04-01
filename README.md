# Person-Tracking-with-FaceID

### Requirements:
``` terminal
conda create --name env python=3.9
conda activate env
pip install -r requirements.txt
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
