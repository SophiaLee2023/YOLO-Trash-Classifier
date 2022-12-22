import torch, torchvision, torchaudio, pandas, cv2 as cv, numpy as np

print(f"PyTorch: {torch.__version__}\t TorchVision: {torchvision.__version__}\t TorchAudio: {torchaudio.__version__}\n" +\
      f"CUDA enabled: {torch.cuda.is_available()}\t CuDNN enabled: {torch.backends.cudnn.enabled}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_bbox(image: np.ndarray, bbox_list: list) -> np.ndarray:
      for xmin, ymin, xmax, ymax, conf, _, label in bbox_list:
            r_xmin, r_ymin, r_xmax, r_ymax = [round(val) for val in (xmin, ymin, xmax, ymax)]
            bbox_color: tuple = tuple(map(int, np.random.choice(range(256), size=3))) # random color

            cv.rectangle(image, (r_xmin, r_ymin), (r_xmax, r_ymax), bbox_color, 2)
            cv.putText(image, f"{label}: {round(conf, 4) * 100}%", (r_xmin, r_ymin - 10), 2, 0.7, bbox_color, 1)
      return image

def detect_frame(image) -> tuple: # NOTE: @image can be a file, Path, URL, PIL, OpenCV, numpy, or list technically
      model = torch.hub.load(".", "custom", path="weights/trash_classifier_v1.pt", source="local") 

      df: pandas.DataFrame = model(image).pandas().xyxy[0] # run the image through the model (make an inference)
      bbox_list: list = [tuple(df.loc[row_index]) for row_index in range(len(df.loc[:]))] # NOTE: [(xmin, ymin, xmax, ymax, conf, class_id, label),]

      return (draw_bbox(cv.imread(image), bbox_list), bbox_list)

results: tuple = detect_frame("data/TACO/images/test/20.jpg")

for detection in results[1]: # iterates through each detected object
      print(detection)
      
cv.imshow("twitch.tv/hushVALO", results[0])
cv.waitKey(0)