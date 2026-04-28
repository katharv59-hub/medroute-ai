from ultralytics import YOLO

model = YOLO(r"runs\detect\runs\detect\ambulance_model_v2-2\weights\best.pt")
results = model(r"TEST_699.mp4", conf=0.3, show=True)