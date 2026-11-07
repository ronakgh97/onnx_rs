//! Onnx operator specifications and attributes
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttributesDataKind {
    F32,
    I64,
    Str,
    Ints,
}

impl AttributesDataKind {
    pub fn name(self) -> &'static str {
        match self {
            AttributesDataKind::F32 => "f32",
            AttributesDataKind::I64 => "i64",
            AttributesDataKind::Str => "string",
            AttributesDataKind::Ints => "i64 list",
        }
    }
}

#[derive(Clone, Copy)]
pub struct OperatorSpec {
    pub min_opset: i64,
    pub arity_in: (usize, usize),
    pub arity_out: (usize, usize),
    pub attrs: &'static [(&'static str, AttributesDataKind)],
}

#[rustfmt::skip]
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Operator {
    Gemm, MatMul, Conv, MaxPool,
    AveragePool, Relu, Sigmoid, Clip,
    DepthToSpace, Flatten, Transpose, Softmax,
    Add, Sub, Mul, Div,
}

impl Operator {
    pub fn lookup(name: &str) -> Option<Operator> {
        let op = match name {
            "Gemm" => Operator::Gemm,
            "MatMul" => Operator::MatMul,
            "Conv" => Operator::Conv,
            "MaxPool" => Operator::MaxPool,
            "AveragePool" => Operator::AveragePool,
            "Relu" => Operator::Relu,
            "Sigmoid" => Operator::Sigmoid,
            "Clip" => Operator::Clip,
            "DepthToSpace" => Operator::DepthToSpace,
            "Flatten" => Operator::Flatten,
            "Transpose" => Operator::Transpose,
            "Add" => Operator::Add,
            "Sub" => Operator::Sub,
            "Mul" => Operator::Mul,
            "Div" => Operator::Div,
            "Softmax" => Operator::Softmax,
            _ => return None,
        };
        Some(op)
    }

    /// Stable name for reports and the pending-kernels list.
    pub fn static_name(self) -> &'static str {
        match self {
            Operator::Gemm => "Gemm",
            Operator::MatMul => "MatMul",
            Operator::Conv => "Conv",
            Operator::MaxPool => "MaxPool",
            Operator::AveragePool => "AveragePool",
            Operator::Relu => "Relu",
            Operator::Sigmoid => "Sigmoid",
            Operator::Clip => "Clip",
            Operator::DepthToSpace => "DepthToSpace",
            Operator::Flatten => "Flatten",
            Operator::Transpose => "Transpose",
            Operator::Add => "Add",
            Operator::Sub => "Sub",
            Operator::Mul => "Mul",
            Operator::Div => "Div",
            Operator::Softmax => "Softmax",
        }
    }

    pub fn spec(self) -> OperatorSpec {
        match self {
            // 13 = C-broadcast rule finalized; alpha/beta/transA/transB
            Operator::Gemm => OperatorSpec {
                min_opset: 13,
                arity_in: (2, 3),
                arity_out: (1, 1),
                attrs: &[
                    ("alpha", AttributesDataKind::F32),
                    ("beta", AttributesDataKind::F32),
                    ("transA", AttributesDataKind::I64),
                    ("transB", AttributesDataKind::I64),
                ],
            },
            // 13 = "since version 13" edition of the definition. No attrs
            Operator::MatMul => OperatorSpec {
                min_opset: 13,
                arity_in: (2, 2),
                arity_out: (1, 1),
                attrs: &[],
            },
            // 11 = pads/strides/dilations semantics settled, group attr present
            Operator::Conv => OperatorSpec {
                min_opset: 11,
                arity_in: (2, 3),
                arity_out: (1, 1),
                attrs: &[
                    ("auto_pad", AttributesDataKind::Str),
                    ("dilations", AttributesDataKind::Ints),
                    ("group", AttributesDataKind::I64),
                    ("kernel_shape", AttributesDataKind::Ints),
                    ("pads", AttributesDataKind::Ints),
                    ("strides", AttributesDataKind::Ints),
                ],
            },
            // 12 = kernel_shape/strides/pads back as attributes; indices out is
            // the optional 2nd output
            Operator::MaxPool => OperatorSpec {
                min_opset: 12,
                arity_in: (1, 1),
                arity_out: (1, 2),
                attrs: &[
                    ("auto_pad", AttributesDataKind::Str),
                    ("ceil_mode", AttributesDataKind::I64),
                    ("count_include_pad", AttributesDataKind::I64),
                    ("kernel_shape", AttributesDataKind::Ints),
                    ("pads", AttributesDataKind::Ints),
                    ("strides", AttributesDataKind::Ints),
                    ("storage_order", AttributesDataKind::I64),
                ],
            },
            Operator::AveragePool => OperatorSpec {
                min_opset: 11,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[
                    ("auto_pad", AttributesDataKind::Str),
                    ("ceil_mode", AttributesDataKind::I64),
                    ("count_include_pad", AttributesDataKind::I64),
                    ("kernel_shape", AttributesDataKind::Ints),
                    ("pads", AttributesDataKind::Ints),
                    ("strides", AttributesDataKind::Ints),
                ],
            },
            // 13 = one stable activation edition; no attrs
            Operator::Relu => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[],
            },
            Operator::Sigmoid => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[],
            },
            // 11 = min/max are INPUTS now (old versions carried attributes)
            Operator::Clip => OperatorSpec {
                min_opset: 11,
                arity_in: (1, 3),
                arity_out: (1, 1),
                attrs: &[],
            },
            // 11 = mode DCR|CRD settled
            Operator::DepthToSpace => OperatorSpec {
                min_opset: 11,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[
                    ("blocksize", AttributesDataKind::I64),
                    ("mode", AttributesDataKind::Str),
                ],
            },
            Operator::Flatten => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[("axis", AttributesDataKind::I64)],
            },
            // 14 = allowzero attribute added
            Operator::Transpose => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[("perm", AttributesDataKind::Ints)],
            },
            // 14 = current broadcast/allowzero edition
            Operator::Add => OperatorSpec {
                min_opset: 14,
                arity_in: (2, 2),
                arity_out: (1, 1),
                attrs: &[],
            },
            Operator::Sub => OperatorSpec {
                min_opset: 14,
                arity_in: (2, 2),
                arity_out: (1, 1),
                attrs: &[],
            },
            Operator::Mul => OperatorSpec {
                min_opset: 14,
                arity_in: (2, 2),
                arity_out: (1, 1),
                attrs: &[("allowzero", AttributesDataKind::I64)],
            },
            Operator::Div => OperatorSpec {
                min_opset: 14,
                arity_in: (2, 2),
                arity_out: (1, 1),
                attrs: &[("allowzero", AttributesDataKind::I64)],
            },
            // 13 = negative-axis semantics settled
            Operator::Softmax => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[("axis", AttributesDataKind::I64)],
            },
        }
    }
}
