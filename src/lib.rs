mod ops;
mod proto;
mod tensor;

pub use crate::ops::{AttributesDataKind, AttributesDataValue, Operator};
use crate::proto::{ModelProto, ValueInfoProto};
pub use crate::tensor::{Tensor, TensorSpec};
use anyhow::{Result, anyhow};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Operator,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<(&'static str, AttributesDataValue)>,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub ir_version: i64,
    pub opset: i64,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub initializers: Vec<Tensor>,
    pub nodes: Vec<Node>,
}

impl Model {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        decode_model(bytes)
    }
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        decode_model(&std::fs::read(path)?)
    }
}

fn decode_model(bytes: &[u8]) -> Result<Model> {
    let proto = match ModelProto::decode(bytes) {
        Ok(m) => m,
        Err(e) => {
            return Err(anyhow!("protobuf decode failed: {}", e));
        }
    };

    // validate ir_version
    let ir_version = match proto.ir_version {
        Some(v) if (3..=13).contains(&v) => v,
        Some(v) => {
            return Err(anyhow!("ir_version {v} invalid, must be 3..=13"));
        }
        None => {
            return Err(anyhow!("ir_version missing, must be 3..=13"));
        }
    };

    // validate opset_import, pick the ONE core version (domain "" or "ai.onnx").
    let mut core_opset: Option<i64> = None;
    for os in &proto.opset_import {
        if os.domain.is_empty() || os.domain == "ai.onnx" {
            if core_opset.is_some() {
                return Err(anyhow!("core opset 'ai.onnx' imported more than once"));
            }
            core_opset = Some(os.version);
        } else {
            return Err(anyhow!(
                "opset domain '{}': runtime implements 'ai.onnx' core ops only",
                os.domain
            ));
        }
    }
    let opset = match core_opset {
        Some(v) if (1..=23).contains(&v) => v,
        Some(v) => {
            return Err(anyhow!(
                "opset {v}: runtime targets opset 1..=23, model declares {v}"
            ));
        }
        None => {
            return Err(anyhow!("opset 'ai.onnx' missing, must declare 1..=23"));
        }
    };

    // validate graph
    let Some(graph) = &proto.graph else {
        return Err(anyhow!("graph missing from model"));
    };

    let mut defined: HashSet<String> = HashSet::default();
    let mut producers: HashMap<String, usize> = HashMap::default();

    let mut init_names: HashSet<String> = HashSet::default();
    let mut initializers: Vec<Tensor> = Vec::with_capacity(graph.initializer.len());
    for (i, tensor) in graph.initializer.iter().enumerate() {
        if tensor.name.is_empty() {
            return Err(anyhow!("initializer #{i} has no name"));
        }
        if !init_names.insert(tensor.name.clone()) {
            return Err(anyhow!("initializer: '{}' is a duplicate", tensor.name));
        }
        // initializer must be f32 only
        if tensor.data_type != 1 {
            return Err(anyhow!("initializer: '{}' is not f32 dtype", tensor.name));
        }

        // check negative dim & numel overflows
        let mut numel: u64 = 1;
        for (di, &d) in tensor.dims.iter().enumerate() {
            if d < 0 {
                return Err(anyhow!(
                    "initializer: '{}' encountered negative dim value: {d} at index dim[{di}]",
                    tensor.name
                ));
            }
            numel = numel.checked_mul(d as u64).ok_or_else(|| {
                anyhow!("initializer: '{}' num of elements overflows", tensor.name)
            })?;
        }

        // collect storage field sources
        let sources: u64 = [
            !tensor.raw_data.is_empty(),
            !tensor.float_data.is_empty(),
            !tensor.int32_data.is_empty(),
            !tensor.int64_data.is_empty(),
        ]
        .into_iter()
        .map(|b| b as u64)
        .sum();

        // initializer must have exactly one storage field
        match sources {
            0 => {
                return Err(anyhow!(
                    "initializer: '{}' {numel} elements, but no data, external storage is not supported",
                    tensor.name
                ));
            }
            1 => {}
            _ => {
                return Err(anyhow!(
                    "initializer: '{}' more than one storage field",
                    tensor.name
                ));
            }
        }

        // initializer must not have int32/int64 data
        if !tensor.int32_data.is_empty() || !tensor.int64_data.is_empty() {
            return Err(anyhow!(
                "initializer: '{}' data in int32/int64 field, only f32 is supported",
                tensor.name
            ));
        }

        // create tensor from raw data, if available
        let tensor = if !tensor.raw_data.is_empty() {
            let expected_size = numel * 4;
            if tensor.raw_data.len() as u64 != expected_size {
                return Err(anyhow!(
                    "initializer: '{}' found {} bytes, tensor shape needs {expected_size}",
                    tensor.name,
                    tensor.raw_data.len()
                ));
            }
            Tensor::create(tensor.name.clone(), tensor.dims.clone(), unsafe {
                raw_as_f32(&tensor.raw_data)
            })
        } else {
            if tensor.float_data.len() as u64 != numel {
                return Err(anyhow!(
                    "initializer: '{}' found {} f32 values, shape needs {numel}",
                    tensor.name,
                    tensor.float_data.len()
                ));
            }
            Tensor::create(
                tensor.name.clone(),
                tensor.dims.clone(),
                tensor.float_data.clone(),
            )
        };

        defined.insert(tensor.name.to_string());
        initializers.push(tensor);
    }

    // validate & collect graph inputs
    let mut inputs: Vec<TensorSpec> = Vec::with_capacity(graph.input.len());
    for (i, vi) in graph.input.iter().enumerate() {
        if vi.name.is_empty() {
            return Err(anyhow!("graph input #{i} has no name"));
        }
        // weight also listed in graph.input
        // the initializer was already validated
        if init_names.contains(vi.name.as_str()) {
            defined.insert(vi.name.clone());
            continue;
        }
        let value = read_declared_value(vi, "graph input")?;
        defined.insert(value.name.to_string());
        inputs.push(value);
    }

    // validate & collect graph outputs
    let mut outputs: Vec<TensorSpec> = Vec::with_capacity(graph.output.len());
    for vo in &graph.output {
        if vo.name.is_empty() {
            return Err(anyhow!("graph output has no name"));
        }
        let value = read_declared_value(vo, "graph output")?;
        outputs.push(value);
    }

    // validate & collect graph nodes
    let mut nodes: Vec<Node> = Vec::with_capacity(graph.node.len());
    for (i, np) in graph.node.iter().enumerate() {
        let label = if np.name.is_empty() {
            return Err(anyhow!("node #{i} has no name"));
        } else {
            np.name.clone()
        };

        let Some(op) = Operator::lookup(&np.op_type) else {
            return Err(anyhow!("{label}: op '{}' is not supported", np.op_type));
        };
        if !np.domain.is_empty() && np.domain != "ai.onnx" {
            return Err(anyhow!("{label}: domain '{}' not core", np.domain));
        }
        let spec = op.spec();
        if opset < spec.min_opset {
            return Err(anyhow!(
                "{label}: '{}' needs opset ≥ {}, model declares {opset}",
                op.static_name(),
                spec.min_opset
            ));
        }
        if np.input.len() < spec.arity_in.0 || np.input.len() > spec.arity_in.1 {
            return Err(anyhow!(
                "{label}: '{}' wants {}..{} inputs, got {}",
                op.static_name(),
                spec.arity_in.0,
                spec.arity_in.1,
                np.input.len()
            ));
        }
        if np.output.len() < spec.arity_out.0 || np.output.len() > spec.arity_out.1 {
            return Err(anyhow!(
                "{label}: '{}' wants {}..{} outputs, got {}",
                op.static_name(),
                spec.arity_out.0,
                spec.arity_out.1,
                np.output.len()
            ));
        }

        // validate & collect node attributes
        let mut attrs: Vec<(&'static str, AttributesDataValue)> = vec![];
        for a in &np.attribute {
            let Some(&(name, kind)) = spec.attrs.iter().find(|(n, _)| *n == a.name.as_str()) else {
                return Err(anyhow!(
                    "{label}: attribute '{}' unknown for '{}'",
                    a.name,
                    op.static_name()
                ));
            };
            let attr = match (kind, &a.f, &a.i, &a.s, &a.ints) {
                (AttributesDataKind::F32, Some(f), _, _, _) => AttributesDataValue::F32(*f),
                (AttributesDataKind::I64, _, Some(i), _, _) => AttributesDataValue::I64(*i),
                (AttributesDataKind::Str, _, _, Some(s), _) => match String::from_utf8(s.clone()) {
                    Ok(st) => AttributesDataValue::Str(st),
                    Err(_) => {
                        return Err(anyhow!("{label}: attribute '{name}' is not valid utf8"));
                    }
                },
                (AttributesDataKind::Ints, _, _, _, _) => AttributesDataValue::Ints(a.ints.clone()),
                _ => {
                    return Err(anyhow!(
                        "{label}: attribute '{}' does not carry a {} payload",
                        a.name,
                        kind.name()
                    ));
                }
            };
            attrs.push((name, attr));
        }

        for oname in &np.output {
            if oname.is_empty() {
                return Err(anyhow!("{label}: empty output name"));
            }
            if producers.insert(oname.clone(), i).is_some() {
                return Err(anyhow!("tensor '{oname}': produced by more than one node"));
            }
            defined.insert(oname.to_string());
        }

        nodes.push(Node {
            op,
            name: label,
            inputs: np.input.clone(),
            outputs: np.output.clone(),
            attrs,
        });
    }

    // check for cycles or undefined inputs
    for (i, np) in graph.node.iter().enumerate() {
        for inp in &np.input {
            if inp.is_empty() {
                continue;
            }
            match producers.get(inp) {
                Some(&p) if p >= i => {
                    return Err(anyhow!(
                        "node#{i}: consumes '{inp}' which is produced later,  graph not topologically sorted"
                    ));
                }
                Some(_) => {}
                None if !defined.contains(inp.as_str()) => {
                    return Err(anyhow!(
                        "tensor '{inp}': consumed by node#{i} but never defined"
                    ));
                }
                None => {}
            }
        }
    }

    for vo in &graph.output {
        if !vo.name.is_empty() && !defined.contains(vo.name.as_str()) {
            return Err(anyhow!("graph output '{}' is never produced", vo.name));
        }
    }

    Ok(Model {
        ir_version,
        opset,
        inputs,
        outputs,
        initializers,
        nodes,
    })
}

fn read_declared_value(vi: &ValueInfoProto, field: &str) -> Result<TensorSpec> {
    let name = vi.name.as_str();
    let Some(tt) = vi.type_.as_ref().and_then(|t| t.tensor_type.as_ref()) else {
        return Err(anyhow!("{field} '{name}': no tensor type declared"));
    };
    if tt.elem_type != 1 {
        return Err(anyhow!(
            "{field} '{name}': {}, only f32 is supported",
            dtype_name(tt.elem_type)
        ));
    }
    let Some(shape) = &tt.shape else {
        return Err(anyhow!(
            "{field} '{name}': no shape declared, concrete shapes required"
        ));
    };
    let mut dims = Vec::with_capacity(shape.dim.len());
    for (di, d) in shape.dim.iter().enumerate() {
        match &d.dim_value {
            Some(v) if *v >= 0 => dims.push(*v),
            Some(v) => {
                return Err(anyhow!(
                    "{field} '{name}': encountered negative dim value: {v} at index dim[{di}]"
                ));
            }
            None => {
                return Err(anyhow!(
                    "{field} '{name}': encountered symbolic dim at index dim[{di}], concrete shapes required"
                ));
            }
        }
    }
    Ok(TensorSpec {
        name: Arc::from(name),
        dims: Arc::from(dims),
    })
}

#[inline]
fn dtype_name(code: i32) -> &'static str {
    match code {
        2 => "uint8",
        3 => "int8",
        4 => "uint16",
        5 => "int16",
        6 => "int32",
        7 => "int64",
        8 => "string",
        9 => "bool",
        10 => "float16",
        11 => "double",
        12 => "uint32",
        13 => "uint64",
        14 => "complex64",
        15 => "complex128",
        16 => "bfloat16",
        _ => "unknown",
    }
}

#[inline]
unsafe fn raw_as_f32(bytes: &[u8]) -> Vec<f32> {
    let (full_chunks, _reminder_chunks) = bytes.as_chunks::<4>();
    full_chunks
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
