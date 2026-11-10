mod kernels;
mod ops;
mod proto;
mod tensor;

// public exports
pub use crate::ops::{AttributesDataValue, Operator, OperatorSpec};
pub use crate::tensor::{Tensor, TensorSpec};

use crate::ops::AttributesDataKind;
use crate::proto::{ModelProto, ValueInfoProto};
use anyhow::{Result, anyhow};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Represents a Node in the ONNX graph,
/// which is an operator with its inputs, outputs, and attributes
#[derive(Clone, Debug)]
pub struct Node {
    pub op: Operator,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<(String, AttributesDataValue)>,
}

/// Represents an ONNX model, which is a directed acyclic graph of nodes (operators)
#[derive(Clone, Debug)]
pub struct Model {
    ir_version: i64,
    opset: i64,
    initializers: Vec<Tensor>,
    nodes: Vec<Node>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl Model {
    /// Create runnable model from slice of bytes
    pub fn from_bytes(bytes: &Arc<[u8]>) -> Result<Self> {
        decode_model(bytes)
    }
    /// Create runnable model from .onnx file path
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        decode_model(&std::fs::read(path)?)
    }
    /// Run 1-input/1-output graph with the given input tensor, returning the output tensor
    pub fn run_single(&self, data: &Tensor) -> Result<Tensor> {
        if self.inputs.len() != 1 || self.outputs.len() != 1 {
            return Err(anyhow!(
                "run_single needs a 1-input/1-output graph, got {}/{}",
                self.inputs.len(),
                self.outputs.len()
            ));
        }
        let mut feed: HashMap<String, Tensor> = HashMap::with_capacity(64);
        feed.insert(self.inputs[0].name.to_string(), data.clone());
        let mut out = self.run(&feed)?;
        Ok(out.remove(0))
    }
    /// Run graph with the given feed of input tensors, returning the output tensors
    pub fn run(&self, feed: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        // validate feed, every declared input is provided, and no extra inputs are given
        for spec in &self.inputs {
            let Some(t) = feed.get(spec.name.as_ref()) else {
                return Err(anyhow!("feed: missing input '{}'", spec.name));
            };
            if t.dims != spec.dims {
                return Err(anyhow!(
                    "input '{}': got dims {:?}, graph declares {:?}",
                    spec.name,
                    t.dims,
                    spec.dims
                ));
            }
        }
        for name in feed.keys() {
            if !self.inputs.iter().any(|s| s.name.as_ref() == name) {
                return Err(anyhow!("feed: unexpected input '{name}'"));
            }
        }

        // build scope, initializers + feed
        let mut scope: HashMap<&str, Tensor> = HashMap::with_capacity(64);
        for t in &self.initializers {
            scope.insert(t.name.as_ref(), t.clone());
        }
        for (name, t) in feed {
            scope.insert(name.as_str(), t.clone());
        }

        // walk the graph in topological order,
        // compute each node's output(s) and bind them to scope
        for node in &self.nodes {
            let mut slots: Vec<Option<&Tensor>> = Vec::with_capacity(node.inputs.len());
            for name in &node.inputs {
                if name.is_empty() {
                    slots.push(None);
                    continue;
                }
                let t = scope
                    .get(name.as_str())
                    .ok_or_else(|| anyhow!("{}: input '{}' is not in scope", node.name, name))?;
                slots.push(Some(t));
            }
            // TODO; some ops have multiple outputs, need to handle them
            let Some(oname) = node.outputs.first() else {
                return Err(anyhow!("{}: node has zero outputs", node.name));
            };
            let out = node.op.compute(oname.as_str(), &slots, &node.attrs)?;
            scope.insert(oname.as_str(), out);
        }

        // graph is done, collect outputs from scope
        let mut produced = Vec::with_capacity(self.outputs.len());
        for spec in &self.outputs {
            let Some(t) = scope.get(spec.name.as_ref()) else {
                return Err(anyhow!("output '{}' was never produced", spec.name));
            };
            if t.dims != spec.dims {
                return Err(anyhow!(
                    "output '{}': got dims {:?}, graph declares {:?}",
                    spec.name,
                    t.dims,
                    spec.dims
                ));
            }
            produced.push(t.clone());
        }
        Ok(produced)
    }
    /// Get the IR version of the model
    pub fn ir_version(&self) -> i64 {
        self.ir_version
    }
    /// Get the opset of the model
    pub fn opset(&self) -> i64 {
        self.opset
    }
    /// Get the initializers of the model
    pub fn initializers(&self) -> &[Tensor] {
        &self.initializers
    }
    /// Get the inputs and outputs of the model
    pub fn inputs_outputs(&self) -> (&[TensorSpec], &[TensorSpec]) {
        (&self.inputs, &self.outputs)
    }
    /// Get all the nodes in the model graph
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    /// Get the operators used in the model
    pub fn operators(&self) -> Vec<Operator> {
        let mut ops: Vec<Operator> = self.nodes.iter().map(|n| n.op).collect();
        ops.sort_unstable();
        ops.dedup();
        ops
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

    let mut defined: HashSet<String> = HashSet::with_capacity(64);
    let mut producers: HashMap<String, usize> = HashMap::with_capacity(64);

    let mut init_names: HashSet<String> = HashSet::with_capacity(64);
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
            return Err(anyhow!(
                "initializer: '{}' must be f32 dtype, got {}",
                tensor.name,
                dtype_name(tensor.data_type)
            ));
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

        // allow empty tensors, for god’s sake,
        // if this creating bad cases to handle, I will lose my shit
        if numel == 0 {
            defined.insert(tensor.name.clone());
            initializers.push(Tensor::create(&tensor.name, &tensor.dims, &[])?);
            continue;
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
            let expected_size = numel
                .checked_mul(4)
                .ok_or_else(|| anyhow!("initializer: '{}' byte size overflows", tensor.name))?;
            if tensor.raw_data.len() as u64 != expected_size {
                return Err(anyhow!(
                    "initializer: '{}' found {} bytes, tensor shape needs {expected_size}",
                    tensor.name,
                    tensor.raw_data.len()
                ));
            }
            Tensor::create(&tensor.name, &tensor.dims, &bytes_to_f32(&tensor.raw_data)?)?
        } else {
            if tensor.float_data.len() as u64 != numel {
                return Err(anyhow!(
                    "initializer: '{}' found {} f32 values, shape needs {numel}",
                    tensor.name,
                    tensor.float_data.len()
                ));
            }
            Tensor::create(&tensor.name, &tensor.dims, &tensor.float_data)?
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
            if !defined.insert(vi.name.clone()) {
                return Err(anyhow!("graph input '{}': duplicate name", vi.name));
            }
            continue;
        }
        let value = read_tensor_spec(vi, "graph input")?;
        if !defined.insert(value.name.to_string()) {
            return Err(anyhow!("graph input '{}': duplicate name", vi.name));
        }
        inputs.push(value);
    }

    // validate & collect graph outputs
    let mut outputs: Vec<TensorSpec> = Vec::with_capacity(graph.output.len());
    for vo in &graph.output {
        if vo.name.is_empty() {
            return Err(anyhow!("graph output has no name"));
        }
        let value = read_tensor_spec(vo, "graph output")?;
        outputs.push(value);
    }

    // validate & collect graph nodes
    let mut nodes: Vec<Node> = Vec::with_capacity(graph.node.len());
    for (i, np) in graph.node.iter().enumerate() {
        let label = if np.name.is_empty() {
            // don't error on empty name,
            // hope it's just a placeholder, not some weird case I need to handle
            format!("node#{i}")
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
                op.static_str(),
                spec.min_opset
            ));
        }
        if np.input.len() < spec.arity_in.0 || np.input.len() > spec.arity_in.1 {
            return Err(anyhow!(
                "{label}: '{}' wants min={}..max={} inputs, got {}",
                op.static_str(),
                spec.arity_in.0,
                spec.arity_in.1,
                np.input.len()
            ));
        }
        if np.output.len() < spec.arity_out.0 || np.output.len() > spec.arity_out.1 {
            return Err(anyhow!(
                "{label}: '{}' wants min={}..max={} outputs, got {}",
                op.static_str(),
                spec.arity_out.0,
                spec.arity_out.1,
                np.output.len()
            ));
        }

        // validate & collect node attributes
        let mut attrs: Vec<(String, AttributesDataValue)> = Vec::with_capacity(np.output.len());
        for a in &np.attribute {
            let Some(&(name, kind)) = spec.attrs.iter().find(|(n, _)| *n == a.name.as_str()) else {
                return Err(anyhow!(
                    "{label}: attribute '{}' unknown for '{}'",
                    a.name,
                    op.static_str()
                ));
            };

            // validate attribute type
            let got = a
                .type_
                .and_then(AttributesDataKind::proto_name_from_proto_code);
            let expected_type = kind.proto_code();
            if a.type_ != Some(expected_type) {
                return Err(anyhow!(
                    "{label}: attribute '{name}' expects type {}, got {:?}",
                    kind.proto_name(),
                    got
                ));
            }

            // check attr value is present and matches expected kind
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
            attrs.push((name.to_string(), attr));
        }

        for oname in &np.output {
            if oname.is_empty() {
                continue;
            }
            if producers.insert(oname.clone(), i).is_some() {
                return Err(anyhow!("tensor '{oname}': produced by more than one node"));
            }
            if !defined.insert(oname.to_string()) {
                return Err(anyhow!(
                    "{label}: output '{oname}' reuses an initializer or graph input name"
                ));
            }
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

#[inline]
fn read_tensor_spec(value_proto: &ValueInfoProto, field: &str) -> Result<TensorSpec> {
    let name = value_proto.name.as_str();
    let Some(tt) = value_proto
        .type_
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
    else {
        return Err(anyhow!("{field} '{name}': no tensor type declared"));
    };
    // proto_code for f32 is 1
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
    // TODO; handle symbolic dims, for now require concrete shapes
    // rank-0 tensors are fucking valid
    // if shape.dim.is_empty() {
    //     return Err(anyhow!("{field} '{name}': shape declared, but rank is 0"));
    // }
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
fn dtype_name(proto_code: i32) -> &'static str {
    match proto_code {
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
fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(anyhow!(
            "initializer: raw_data has {} bytes, cannot be converted to f32",
            bytes.len()
        ));
    }

    let (bytes_chunks, _) = bytes.as_chunks::<4>();
    let mut floats = Vec::with_capacity(bytes_chunks.len());
    floats.extend(
        bytes_chunks
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
    );

    Ok(floats)
}
