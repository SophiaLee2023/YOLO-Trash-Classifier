import os

def run_in_terminal(cmd_list: list) -> None:
    for cmd_str in cmd_list:
        os.system(cmd_str)

run_in_terminal([
    "pip install -r requirements.txt",
    "python data/TACO/download_script.py",
    "python train.py --img 640 --batch 32 --epochs 10 --data TACO.yaml --weights yolov5s.pt --name trash_classifier --cache"
])