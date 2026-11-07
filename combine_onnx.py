import onnx
model = onnx.load("srgan_onnx/srgan.onnx")
onnx.save(model, "srgan_onnx/srgan_single.onnx")