# YOLO v7 TACO Trash Classifier

## Set-up Commands

``` shell
# downloads required packages
pip install -r requirements.txt

# downloads TACO images (NOTE: FCPS wifi results in a connection error) and generates annotations
python data/TACO/download_script.py

# optional: reinstall the GPU-compatible version of PyTorch to make training faster
pip uninstall torch
pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio===0.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

## Train and Detect on Coco128

``` shell
# training command
python train.py --epochs 3 --data data/coco128.yaml --weights weights/yolov5s.pt --name coco128_classifier --cache disk

# detection command from two different sources (check results in runs/detect/coco128_classifier)
python detect.py --weights runs/train/coco128_classifier/weights/best.pt --source data/coco128/images/train2017 --name coco128_classifier
python detect.py --img 1280 --weights weights/yolov5s.pt --source 0 --name webcam_recording
```
If using pre-trained weights, change the file path after "--weights" to the location of the downloaded ".pt" file.

## Train and Detect on TACO

``` shell
# training command
python train.py --epochs 100 --data data/TACO.yaml --cfg models/yolov5s_nc60.yaml --weights weights/yolov5s.pt --name trash_classifier --cache disk

# detection command from two different sources (check results in runs/detect/trash_classifier)
python detect.py --img 1280 --weights weights/trash_classifier_v1.pt --source data/TACO/images/test --conf 0.1 --name trash_classifier
python detect.py --img 1280 --weights weights/trash_classifier_v1.pt --source 0 --conf 0.1 --name webcam_recording
```