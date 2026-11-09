Minimal **onnx runtime** for running `f32-only` models on **x86_64** CPUs ⚡

Note:

- Runtime is very constrained, only few type of models and certain runtime behaviors are supported (thanks to onnx spec
  for making harder)
- Model parser is quite weak and complex to implement and needs to have a reasonably simple graph (send PR or help me)
- Nothing is assumed, anything "alien" to our scope is rejected and errors instead of silently ignoring

> The goal is that "`If model passes the checks during loading, it should run without any errors`"

Benchmarks and correctness (running [srgan](https://aihub.qualcomm.com/compute/models/srgan), 1.55M params):

| Runtime | Time     | Input                    | Output                     |
|---------|----------|--------------------------|----------------------------|
| ORT     | 6.90s    | ![img](image_sample.png) | ![img](ort_output.png)     |
| ONNX_RS | 335.79ms | ![img](image_sample.png) | ![img](onnx_rs_output.png) |
