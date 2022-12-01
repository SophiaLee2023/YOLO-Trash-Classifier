import os, torch, torchvision, torchaudio

def run_in_terminal(command_list: list) -> None:
    for command_str in command_list:
        os.system(command_str)

def install_requirements(GPU_integration: bool = False) -> None:
    run_in_terminal([
        "pip install -r requirements.txt", # downloads YOLO v5 required packages
        "python data/TACO/download_script.py" # downloads TACO images (NOTE: FCPS wifi results in a connection error) and generates YOLO v5 annotations
    ])

    if GPU_integration: # NOTE: for GPU-PyTorch integration (optional), install CUDA-11.7.0 and cuDNN-8.6.0
        run_in_terminal([ 
            "pip uninstall torch", # remove torch-1.13.0+cpu (default installation is CPU only)
            "pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio===0.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html",
        ])

# install_requirements()

print(f"PyTorch: {torch.__version__}\t TorchVision: {torchvision.__version__}\t TorchAudio: {torchaudio.__version__}\n" +\
      f"CUDA enabled: {torch.cuda.is_available()}\t CuDNN enabled: {torch.backends.cudnn.enabled}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_run(epochs: int, data: str, cfg: str, weights: str, source: str, name: str, img: int, conf: float) -> None:
    run_in_terminal([
        f"python train.py --epochs {epochs} --img {img} --data {data} --cfg {cfg} --weights {weights} --name {name} --cache disk",
        f"python detect.py --img {img} --weights runs/train/{name}/weights/best.pt --conf {conf} --source {source} --name {name}"
    ])

# train_and_run(3, "data/coco128.yaml", "", "weights/yolov5s.pt", "data/coco128/images/train2017", "coco128_classifier", 640, 0.25)

# run_in_terminal(["python detect.py --weights weights/yolov5s.pt --source data/TACO/images/"]) 
train_and_run(100, "data/TACO.yaml", "models/yolov5s_nc60.yaml", "weights/yolov5s.pt", "data/TACO/images/test", "trash_classifier", 640, 0.1)