import torch
from blazeface import BlazeFace  # from your blazeface.py

# Load model
net = BlazeFace()
net.load_weights("blazeface.pth")
net.load_anchors("anchors.npy")
net.eval()

# Dummy input (BlazeFace expects 128x128 RGB images)
dummy_input = torch.randn(1, 3, 128, 128)

# Export to ONNX
torch.onnx.export(
    net,
    dummy_input,
    "blazeface.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("✅ Saved blazeface.onnx")
