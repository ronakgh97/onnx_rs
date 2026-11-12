Minimal **onnx runtime** for running `f32 precision` models on **x86_64** CPUs ⚡

Notes:

- Runtime is very constrained, only few type of models and certain runtime behaviors are supported (thanks to onnx spec
  for making this harder)
- Model parser is quite weak and complex to implement and needs to have a reasonably simple graph
- Nothing is assumed, anything "alien" to this runtime's scope is rejected and panics instead of silently ignoring or
  assuming, which may lead to bad undesired outputs
- Operators correctness tests are covered yet, so expect some undesired behaviors/outputs on sophisticated operators
  used by the model

> The goal is that:
> "`If model passes the checks during loading, it should run without any silent errors or corrupted outputs`"

Benchmarks and correctness:

Running [SRGAN](https://aihub.qualcomm.com/compute/models/srgan) 5.92 MB

| Runtime | Time     | Input                                       | Output                                              |
|---------|----------|---------------------------------------------|-----------------------------------------------------|
| ONNX_RS | 4.54s    | <img src="images/sample_1.png" width="512"> | <img src="images/srgan_onnx_rs_x4.png" width="512"> |
| ORT     | 216.34ms | <img src="images/sample_1.png" width="512"> | <img src="images/srgan_ort_x4.png" width="512">     |

Running [4xPurePhoto-Span](https://openmodeldb.info/models/4x-PurePhoto-span) 1.72 MB

| Runtime | Time     | Input                                       | Output                                                       |
|---------|----------|---------------------------------------------|--------------------------------------------------------------|
| ONNX_RS | 814.22ms | <img src="images/sample_2.png" width="512"> | <img src="images/purephoto_span_onnx_rs_x4.png" width="512"> |
| ORT     | 100.29ms | <img src="images/sample_2.png" width="512"> | <img src="images/purephoto_span_ort_x4.png" width="512">     |
