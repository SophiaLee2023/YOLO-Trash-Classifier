import torch, torchvision, torchaudio, pandas, numpy as np

print(f"PyTorch: {torch.__version__}\t TorchVision: {torchvision.__version__}\t TorchAudio: {torchaudio.__version__}\n" +\
      f"CUDA enabled: {torch.cuda.is_available()}\t CuDNN enabled: {torch.backends.cudnn.enabled}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_frame(image) -> list: # NOTE: @image can be a file, Path, URL, PIL, OpenCV, numpy, or list
      model = torch.hub.load(".", "custom", path="weights/trash_classifier_v1.pt", source="local") 

      df: pandas.DataFrame = model(image).pandas().xyxy[0] # run the image through the model (make an inference)
      return [tuple(df.loc[row_index]) for row_index in range(len(df.loc[:]))] # NOTE: [(xmin, ymin, xmax, ymax, confidence, class_id, label),]

results: list = detect_frame("data/TACO/images/test/20.jpg")

for detection in results: # iterates through each detected object
      print(detection)