import threading
import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name])


script1_process = threading.Thread(target=run_script, args=("IN.py",))
script2_process = threading.Thread(target=run_script, args=("OUT.py",))

script1_process.start()
script2_process.start()

script1_process.join()
script2_process.join()