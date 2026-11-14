//! Onnx operator specifications and attributes
use anyhow::Result;
use anyhow::anyhow;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttributesDataKind {
    F32 = 1,
    I64 = 2,
    Str = 3,
    // ... some crap that we don't support yet
    // F32s = 6,
    Ints = 7,
    // Strs = 8,
    // ... more crap we don't want
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum AttributesDataValue {
    F32(f32),
    I64(i64),
    Str(String),
    Ints(Vec<i64>),
}

#[allow(unused)]
pub fn take_attr_f32<'s>(
    attrs: &'s [(String, AttributesDataValue)],
    name: &str,
) -> Result<Option<&'s f32>> {
    for (k, v) in attrs {
        if k == name {
            let AttributesDataValue::F32(v) = v else {
                return Err(anyhow!("attribute '{name}': wrong payload kind"));
            };
            return Ok(Some(v));
        }
    }
    Ok(None)
}

#[inline(always)]
pub fn take_attr_i64<'s>(
    attrs: &'s [(String, AttributesDataValue)],
    name: &str,
) -> Result<Option<&'s i64>> {
    for (k, v) in attrs {
        if k == name {
            let AttributesDataValue::I64(v) = v else {
                return Err(anyhow!("attribute '{name}': wrong payload kind"));
            };
            return Ok(Some(v));
        }
    }
    Ok(None)
}

#[inline(always)]
pub fn take_attr_str<'s>(
    attrs: &'s [(String, AttributesDataValue)],
    name: &str,
) -> Result<Option<&'s str>> {
    for (k, v) in attrs {
        if k == name {
            let AttributesDataValue::Str(s) = v else {
                return Err(anyhow!("attribute '{name}': wrong payload kind"));
            };
            return Ok(Some(s));
        }
    }
    Ok(None)
}

#[inline(always)]
pub fn take_attr_i64s<'s>(
    attrs: &'s [(String, AttributesDataValue)],
    name: &str,
) -> Result<Option<&'s [i64]>> {
    for (k, v) in attrs {
        if k == name {
            let AttributesDataValue::Ints(v) = v else {
                return Err(anyhow!("attribute '{name}': wrong payload kind"));
            };
            return Ok(Some(v));
        }
    }
    Ok(None)
}

pub fn take_attr_i64s_n<const N: usize>(
    attrs: &[(String, AttributesDataValue)],
    name: &str,
) -> Result<Option<[i64; N]>> {
    let Some(v) = take_attr_i64s(attrs, name)? else {
        return Ok(None);
    };
    if v.len() != N {
        return Err(anyhow!(
            "attribute '{name}': wants {N} i64s, got {}",
            v.len()
        ));
    }
    let mut out = [0i64; N];
    out.copy_from_slice(v);
    Ok(Some(out))
}

#[rustfmt::skip]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operator {
    Clip, Relu, PRelu,  DepthToSpace,
    Conv, Gemm, MatMul, MaxPool,
    AveragePool, Sigmoid, Flatten, Transpose,
    Softmax, Add, Sub, Mul, Div, 
    Concat, SoftPlus, ReduceMean, Tanh,

}

#[derive(Clone, Copy, Debug)]
pub struct OperatorSpec {
    pub min_opset: i64,
    pub arity_in: (usize, usize),  // (min..max)
    pub arity_out: (usize, usize), // (min..max)
    pub attrs: &'static [(&'static str, AttributesDataKind)],
}

impl Operator {
    #[inline(always)]
    #[rustfmt::skip]
    pub fn static_str(self) -> &'static str {
        match self {
            Operator::Clip => "Clip", Operator::Relu => "Relu", Operator::PRelu => "PRelu", Operator::DepthToSpace => "DepthToSpace",
            Operator::Conv => "Conv", Operator::Gemm => "Gemm", Operator::MatMul => "MatMul", Operator::MaxPool => "MaxPool",
            Operator::AveragePool => "AveragePool", Operator::Sigmoid => "Sigmoid", Operator::Flatten => "Flatten", Operator::Transpose => "Transpose",
            Operator::Softmax => "Softmax", Operator::Add => "Add", Operator::Sub => "Sub", Operator::Mul => "Mul", Operator::Div => "Div",
            Operator::Concat => "Concat", Operator::SoftPlus => "Softplus", Operator::ReduceMean => "ReduceMean", Operator::Tanh => "Tanh"
        }
    }

    #[inline(always)]
    #[rustfmt::skip]
    pub fn lookup(name: &str) -> Option<Operator> {
        let op = match name {
            "Clip" => Operator::Clip, "Relu" => Operator::Relu, "PRelu" => Operator::PRelu, "DepthToSpace" => Operator::DepthToSpace,
            "Conv" => Operator::Conv, "Gemm" => Operator::Gemm, "MatMul" => Operator::MatMul, "MaxPool" => Operator::MaxPool,
            "AveragePool" => Operator::AveragePool, "Sigmoid" => Operator::Sigmoid, "Flatten" => Operator::Flatten, "Transpose" => Operator::Transpose,
            "Softmax" => Operator::Softmax, "Add" => Operator::Add, "Sub" => Operator::Sub, "Mul" => Operator::Mul, "Div" => Operator::Div,
            "Concat" => Operator::Concat, "Softplus" => Operator::SoftPlus, "ReduceMean" => Operator::ReduceMean, "Tanh" => Operator::Tanh,
            _ => return None,
        };
        Some(op)
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
            Operator::Concat => OperatorSpec {
                min_opset: 4,
                arity_in: (1, 2147483647),
                arity_out: (1, 1),
                attrs: &[("axis", AttributesDataKind::I64)],
            },
            Operator::SoftPlus => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[],
            },
            Operator::ReduceMean => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[
                    ("axes", AttributesDataKind::Ints),
                    ("keepdims", AttributesDataKind::I64),
                ],
            },
            Operator::Tanh => OperatorSpec {
                min_opset: 13,
                arity_in: (1, 1),
                arity_out: (1, 1),
                attrs: &[],
            },
        }
    }
}
