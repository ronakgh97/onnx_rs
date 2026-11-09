import sys
import onnx
input_path = sys.argv[1]
output_path = sys.argv[2]
model = onnx.load(input_path)
onnx.save(model, output_path)