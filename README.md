Minimal **onnx runtime** for running `f32 precision` models on **x86_64** CPUs ⚡

Notes:

- Runtime is very constrained, only few type of models and certain runtime behaviors are supported (thanks to onnx spec
  for making harder)
- Model parser is quite weak and complex to implement and needs to have a reasonably simple graph
- Nothing is assumed, anything "alien" to this runtime's scope is rejected and panics instead of silently ignoring or
  assuming, which may lead to bad undesired outputs
- Operators correctness tests are covered yet, so expect some undesired behaviors/outputs on sophisticated operators
  used by the model

> The goal is that:
> "`If model passes the checks during loading, it should run without any silent errors or corrupted outputs`"

Benchmarks and correctness (running [srgan](https://aihub.qualcomm.com/compute/models/srgan), 1.55M params):

| Runtime | Time     | Input                                     | Output                                        |
|---------|----------|-------------------------------------------|-----------------------------------------------|
| ONNX_RS | 6.11s    | <img src="images/sample.png" width="512"> | <img src="images/onnx_rs_x4.png" width="512"> |
| ORT     | 248.34ms | <img src="images/sample.png" width="512"> | <img src="images/ort_x4.png" width="512">     |
