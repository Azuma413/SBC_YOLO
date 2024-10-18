from ultralytics import YOLO

model_path = "yolov10s.pt"

# convert to ONNX model
model = YOLO(model_path)
model.export(format='onnx', imgsz=[720, 1280])