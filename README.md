Minimal **onnx runtime** for running `f32 precision` models on **x86_64** CPUs ⚡

Notes:

- Runtime is very constrained, only few type of models and certain runtime behaviors are supported (thanks to onnx spec
  for making harder)
- Model parser is quite weak and complex to implement and needs to have a reasonably simple graph (send PR or help me)
- Nothing is assumed, anything "alien" to our scope is rejected and panics instead of silently ignoring or assuming,
  which may lead to bad undesired outputs

> The goal is that "`If model passes the checks during loading, it should run without any silent errors or corruption`"

Benchmarks and correctness (running [srgan](https://aihub.qualcomm.com/compute/models/srgan), 1.55M params):

| Runtime | Time     | Input                   | Output                     |
|---------|----------|-------------------------|----------------------------|
| ORT     | 6.47s    | ![img](input_image.png) | ![img](ort_output.png)     |
| ONNX_RS | 311.60ms | ![img](input_image.png) | ![img](onnx_rs_output.png) |
