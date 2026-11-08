//! Onnx operator specifications and attributes
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttributesDataKind {
    F32,
    I64,
    Str,
    Ints,
}

#[derive(Clone, Debug)]
pub enum AttributesDataValue {
    F32(f32),
    I64(i64),
    Str(String),
    Ints(Vec<i64>),
}

impl AttributesDataKind {
    fn metadata(&self) -> (&'static str, i32, &'static str) {
        match self {
            AttributesDataKind::F32 => ("f32", 1, "FLOAT"),
            AttributesDataKind::I64 => ("i64", 2, "INT"),
            AttributesDataKind::Str => ("string", 3, "STRING"),
            AttributesDataKind::Ints => ("i64 list", 7, "INTS"),
        }
    }

    #[inline(always)]
    pub fn name(self) -> &'static str {
        self.metadata().0
    }
    #[inline(always)]
    pub fn proto_code(&self) -> i32 {
        self.metadata().1
    }
    #[inline(always)]
    pub fn proto_name(&self) -> &'static str {
        self.metadata().2
    }
    #[inline(always)]
    pub fn proto_name_from_proto_code(code: i32) -> Option<&'static str> {
        match code {
            1 => Some("FLOAT"),
            2 => Some("INT"),
            3 => Some("STRING"),
            7 => Some("INTS"),
            _ => None,
        }
    }
}

#[rustfmt::skip]
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Operator {
    Clip, Relu, PRelu,  DepthToSpace,
    Conv, Gemm, MatMul, MaxPool,
    AveragePool, Sigmoid, Flatten, Transpose,
    Softmax, Add, Sub, Mul, Div,

}

#[derive(Clone, Copy)]
pub struct OperatorSpec {
    pub min_opset: i64,
    pub arity_in: (usize, usize),
    pub arity_out: (usize, usize),
    pub attrs: &'static [(&'static str, AttributesDataKind)],
}

impl Operator {
    #[inline(always)]
    #[rustfmt::skip]
    pub fn lookup(name: &str) -> Option<Operator> {
        let op = match name {
            "Clip" => Operator::Clip, "Relu" => Operator::Relu, "PRelu" => Operator::PRelu, "DepthToSpace" => Operator::DepthToSpace,
            "Conv" => Operator::Conv, "Gemm" => Operator::Gemm, "MatMul" => Operator::MatMul, "MaxPool" => Operator::MaxPool,
            "AveragePool" => Operator::AveragePool, "Sigmoid" => Operator::Sigmoid, "Flatten" => Operator::Flatten, "Transpose" => Operator::Transpose,
            "Softmax" => Operator::Softmax, "Add" => Operator::Add, "Sub" => Operator::Sub, "Mul" => Operator::Mul, "Div" => Operator::Div,
            _ => return None,
        };
        Some(op)
    }

    #[inline(always)]
    #[rustfmt::skip]
    pub fn static_name(self) -> &'static str {
        match self {
            Operator::Clip => "Clip", Operator::Relu => "Relu", Operator::PRelu => "PRelu", Operator::DepthToSpace => "DepthToSpace",
            Operator::Conv => "Conv", Operator::Gemm => "Gemm", Operator::MatMul => "MatMul", Operator::MaxPool => "MaxPool",
            Operator::AveragePool => "AveragePool", Operator::Sigmoid => "Sigmoid", Operator::Flatten => "Flatten", Operator::Transpose => "Transpose",
            Operator::Softmax => "Softmax", Operator::Add => "Add", Operator::Sub => "Sub", Operator::Mul => "Mul", Operator::Div => "Div",
        }
    }

    #[inline(always)]
    pub fn spec(self) -> OperatorSpec {
        match self {
            Operator::Clip => OperatorSpec {
                min_opset: 11,
                arity_in: (1, 3),
                arity_out: (1, 1),
                attrs: &[],
            },
            Operator::Relu => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[],
            },
            Operator::PRelu => OperatorSpec {
                min_opset: 16,
                arity_in: (2, 2),
                arity_out: (1, 1),
                attrs: &[],
            },
            Operator::DepthToSpace => OperatorSpec {
                min_opset: 11,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[
                    ("blocksize", AttributesDataKind::I64),
                    ("mode", AttributesDataKind::Str),
                ],
            },
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
            Operator::MatMul => OperatorSpec {
                min_opset: 13,
                arity_in: (2, 2),
                arity_out: (1, 1),
                attrs: &[],
            },
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
            Operator::Sigmoid => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[],
            },
            Operator::Flatten => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[("axis", AttributesDataKind::I64)],
            },
            Operator::Transpose => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[("perm", AttributesDataKind::Ints)],
            },
            Operator::Softmax => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[("axis", AttributesDataKind::I64)],
            },
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
        }
    }
}
