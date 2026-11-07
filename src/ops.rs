//! Onnx operators and their corresponding implementations
#[rustfmt::skip]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operator {
    Gemm, MatMul, Conv, MaxPool,
    AveragePool, Flatten, Add, Sub,
    Mul, Div, Neg, Relu, Sigmoid,
    Tanh, Erf, Exp, Log,
    Sqrt, Pow, Cast, Shape,
    Reshape, Transpose, Concat, Gather,
    Softmax, Slice, Squeeze, Unsqueeze,
    Constant, ConstantOfShape, Expand, ReduceMean,
    ArgMax, Equal, Less, Greater, Where,
}
