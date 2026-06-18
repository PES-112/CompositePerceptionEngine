from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# 1. Download the standard FP16/FP32 base weights
model_path = hf_hub_download(repo_id="openvision/yolo26-n", filename="model.pt")
model = YOLO(model_path)

# 2. Convert the base model to an INT8 quantized format (e.g., ONNX)
# Passing the data argument calibrates the 8-bit mapping
int8_onnx_path = model.export(format="onnx", int8=True, data="coco8.yaml")

# 3. Load the newly generated 8-bit quantized model
quantized_model = YOLO(int8_onnx_path)

# 4. Run inference using 8-bit integer weights
results = quantized_model("traffic.mp4")