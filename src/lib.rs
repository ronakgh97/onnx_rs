//! A small, pure-Rust ONNX runtime for running supported `f32` models on
//! x86_64 CPUs.
//!
//! Models are loaded from `.onnx` (must contain both graph and initializers)
//! files or encoded bytes, validated while loading,
//! and executed with [`Model::run_multiple`]. The public API keeps the
//! runtime deliberately small: [`Model`] describes and runs a graph, while
//! [`Tensor`] and [`TensorSpec`] represent its data and input/output shapes.
//!
//! The runtime currently supports a focused subset of ONNX operators and
//! model shapes. Unsupported graph features are rejected instead of being
//! silently ignored or producing undesired outputs.
//!
//! Comparison with [ORT](https://docs.rs/ort/latest/ort/), running [SRGAN](https://aihub.qualcomm.com/compute/models/srgan) (5.92 MB) from Qualcomm's AI Hub
//!
//! | Runtime | Input | Output |
//! |---------|-------|--------|
//! | ONNX_RS | <img src="https://raw.githubusercontent.com/ronakgh97/onnx_rs/refs/heads/master/images/sample_1.png" width="512"> | <img src="https://raw.githubusercontent.com/ronakgh97/onnx_rs/refs/heads/master/images/srgan_onnx_rs_x4.png" width="512"> |
//! | ORT     | <img src="https://raw.githubusercontent.com/ronakgh97/onnx_rs/refs/heads/master/images/sample_1.png" width="512"> | <img src="https://raw.githubusercontent.com/ronakgh97/onnx_rs/refs/heads/master/images/srgan_ort_x4.png" width="512">     |
//!
//! ```no_run
//! use onnx_rs::{Model, Tensor};
//! fn main() {
//!     let model = Model::from_file("path/to/model.onnx").unwrap();
//!     let input = Tensor::create("input_name", &[1, 3, 128, 128], vec![0f32; 1*3*128*128]).unwrap();
//!     let output = model.run_single(input).unwrap();
//!     print!("output tensor dims: {:?}", output.dims);
//! }
//! ```
//!

mod kernels;
mod ops;
mod proto;
mod tensor;

// public only get tensor utils and model runner, no abstraction leaking
pub use crate::tensor::{Tensor, TensorSpec};

use crate::ops::AttributesDataKind;
use crate::ops::{AttributesDataValue, Operator};
use crate::proto::{ModelProto, TensorProto, ValueInfoProto};
use anyhow::{Result, anyhow};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Represents a Node in the ONNX graph,
/// which is an operator with its inputs, outputs, and attributes
#[repr(C)]
#[derive(Clone)]
pub struct Node {
    name: String,                                // node_name
    op: Operator,                                // operator_type
    inputs: Arc<[String]>,                       // multiple_inputs
    outputs: Arc<[String]>,                      // multiple_outputs
    attrs: Arc<[(String, AttributesDataValue)]>, // list of (attribute_name, attribute_value) associated
}

impl Node {
    /// Name of the node
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Operation type of the node, e.g., "Conv", "Relu", etc.
    pub fn op(&self) -> &'static str {
        self.op.static_str()
    }
    /// Inputs and outputs of the node
    pub fn inputs_outputs(&self) -> (&[String], &[String]) {
        (&self.inputs, &self.outputs)
    }
}

/// Represents an ONNX model, which is a directed acyclic graph of nodes (operators)
#[repr(C)]
#[derive(Clone)]
pub struct Model {
    ir_version: i64,             // for version check
    opset: i64,                  // for compatibility check
    nodes: Arc<[Node]>,          // all nodes in the graph, in topological order
    operators: Arc<[String]>, // all the operators used in the graph, in topological order (deduped)
    initializers: Arc<[Tensor]>, // all weights and biases
    inputs: Arc<[TensorSpec]>, // all the inputs that the graph expects, with their names and shapes
    outputs: Arc<[TensorSpec]>, // all the outputs that the graph produces, with their names and shapes
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
    pub fn run_single(&self, data: Tensor) -> Result<Tensor> {
        if self.inputs.len() != 1 || self.outputs.len() != 1 {
            return Err(anyhow!(
                "run_single needs a 1-input/1-output graph, got {}/{}",
                self.inputs.len(),
                self.outputs.len()
            ));
        }

        if data.dims != self.inputs[0].dims {
            return Err(anyhow!(
                "run_single: input dims {:?} does not match graph declared dims {:?}",
                data.dims,
                self.inputs[0].dims
            ));
        }

        // build scope, initializers + constants + input
        // TODO: this keeps growing
        let mut scope: HashMap<&str, Tensor> =
            HashMap::with_capacity(self.initializers.len() + self.nodes.len() * 2);
        for t in self.initializers.iter() {
            scope.insert(t.name.as_ref(), t.clone());
        }
        scope.insert(self.inputs[0].name.as_ref(), data);

        let mut record_timings = HashMap::with_capacity(self.operators.len());

        // walk & execute the graph in topological order
        Self::run_impl(&self.nodes, &mut scope, &mut record_timings)?;

        let Some(t) = scope.get(self.outputs[0].name.as_ref()) else {
            return Err(anyhow!(
                "output '{}' was never produced",
                self.outputs[0].name
            ));
        };
        if t.dims != self.outputs[0].dims {
            return Err(anyhow!(
                "output '{}': got dims {:?}, graph declares {:?}",
                self.outputs[0].name,
                t.dims,
                self.outputs[0].dims
            ));
        }

        dbg!("timings: {:?}", record_timings);

        Ok(t.clone())
    }
    /// Run graph with the given feed of input tensors, returning the output tensors
    pub fn run_multiple(&self, feed: HashMap<&str, Tensor>) -> Result<Vec<Tensor>> {
        // validate feed, every declared input is provided,
        // and no extra inputs are given
        for spec in self.inputs.iter() {
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
            if !self.inputs.iter().any(|s| s.name.as_ref() == *name) {
                return Err(anyhow!("feed: unexpected input '{name}'"));
            }
        }

        // build scope, initializers + constants + feed
        let mut scope: HashMap<&str, Tensor> =
            HashMap::with_capacity(self.initializers.len() + feed.len() + self.nodes.len() * 2);
        for t in self.initializers.iter() {
            scope.insert(t.name.as_ref(), t.clone());
        }
        for (name, t) in feed {
            scope.insert(name, t);
        }

        let mut record_timings = HashMap::with_capacity(self.operators.len());

        // walk & execute the graph in topological order
        Self::run_impl(&self.nodes, &mut scope, &mut record_timings)?;

        // graph is done, collect outputs from scope
        let mut produced = Vec::with_capacity(self.outputs.len());
        for spec in self.outputs.iter() {
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

        dbg!("timings: {:?}", record_timings);

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
    /// Get all the nodes in the model graph in topological order
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    /// Get the operators used in the model in topological order (deduplicated)
    pub fn operators(&self) -> &[String] {
        &self.operators
    }

    #[inline]
    fn run_impl<'s>(
        nodes: &'s [Node],
        scope: &mut HashMap<&'s str, Tensor>,
        timings: &mut HashMap<&str, f32>,
    ) -> Result<()> {
        // walk the graph in topological order,
        // compute each node's output(s) and bind them to scope
        let mut output_declared: Vec<&str> = Vec::with_capacity(24);
        for node in nodes {
            // collect inputs first, avoids borrow conflict when inserting outputs
            let mut inputs: Vec<Option<&Tensor>> = Vec::with_capacity(node.inputs.len());
            for name in node.inputs.iter() {
                if name.is_empty() {
                    inputs.push(None);
                    continue;
                }
                let t = scope
                    .get(name.as_str())
                    .ok_or_else(|| anyhow!("{}: input '{}' is not in scope", node.name(), name))?;
                inputs.push(Some(t));
            }
            output_declared.clear(); // reuse buffer

            // fill in output_declared with non-empty output names
            for name in node.outputs.iter() {
                if !name.is_empty() {
                    output_declared.push(name.as_str());
                }
            }
            if output_declared.is_empty() {
                return Err(anyhow!("{}: node has zero non-empty outputs", node.name()));
            }
            // dbg!(
            //     "executing node",
            //     node.name(),
            //     node.op(),
            //     node.inputs_outputs(),
            // );

            let start = Instant::now();
            let output_tensor = node.op.compute(&output_declared, &inputs, &node.attrs)?;
            let end = start.elapsed().as_secs_f32();

            // record timings for this operator
            timings
                .entry(node.op())
                .and_modify(|t| *t += end)
                .or_insert(end);

            if output_tensor.len() != output_declared.len() {
                return Err(anyhow!(
                    "{}: operator '{:?}' produced {} outputs, declared {}",
                    node.name(),
                    node.op,
                    output_tensor.len(),
                    output_declared.len()
                ));
            }
            // bind outputs to scope for next nodes to consume
            for (name, out) in output_declared.iter().zip(output_tensor) {
                scope.insert(name, out);
            }
        }
        Ok(())
    }
}

// TODO: Better error messages, more failure context and consistency

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

    // initialize scope tracking structures
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

        // validate & collect initializer into scope, they are available to all nodes
        let tensor = read_tensor(tensor, "initializer", &tensor.name)?;
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
        // validate tensor spec, and add to graph inputs
        let value = read_tensor_spec(vi, "graph input")?;
        if !defined.insert(value.name.to_string()) {
            return Err(anyhow!("graph input '{}': duplicate name", vi.name));
        }
        inputs.push(value);
    }

    // validate & collect graph outputs
    let mut outputs: Vec<TensorSpec> = Vec::with_capacity(graph.output.len());
    let mut output_names: HashSet<String> = HashSet::with_capacity(graph.output.len());
    for vo in &graph.output {
        if vo.name.is_empty() {
            return Err(anyhow!("graph output has no name"));
        }
        if !output_names.insert(vo.name.clone()) {
            return Err(anyhow!("graph output '{}': duplicate name", vo.name));
        }
        // validate tensor spec, and add to graph outputs
        let value = read_tensor_spec(vo, "graph output")?;
        outputs.push(value);
    }

    // validate & collect graph execution nodes and their operators
    let mut nodes: Vec<Node> = Vec::with_capacity(graph.node.len());
    for (i, np) in graph.node.iter().enumerate() {
        let node_name = if np.name.is_empty() {
            // don't error on empty name, onnx allows optional node names
            // TODO: hope it's just a placeholder, not some weird case I don't want to handle
            format!("unknown#{i}")
        } else {
            np.name.clone()
        };

        if !np.domain.is_empty() && np.domain != "ai.onnx" {
            return Err(anyhow!("{node_name}: domain '{}' not core", np.domain));
        }

        // 'Constant' nodes carry a TENSOR attribute, which the
        // operator/attribute system does not model,
        // semantically they are graph initializers that
        // happen to be spelled as nodes, so fold them into the
        // scope at decode time and drop the node entirely
        if np.op_type == "Constant" {
            if !np.input.is_empty() {
                return Err(anyhow!("{node_name}: Constant must have no inputs"));
            }

            let out_name = if let Some(output) = np.output.first() {
                if output.is_empty() {
                    return Err(anyhow!(
                        "{node_name}: Constant must have a non-empty output name"
                    ));
                }
                output
            } else {
                return Err(anyhow!(
                    "{node_name}: Constant must have exactly one non-empty output"
                ));
            };

            // only support tensor value
            let Some(attr) = np.attribute.iter().find(|a| a.name == "value") else {
                return Err(anyhow!(
                    "{node_name}: Constant requires a 'value' attribute"
                ));
            };
            let Some(tp) = attr.t.as_ref() else {
                return Err(anyhow!(
                    "{node_name}: Constant 'value' must be a tensor attribute"
                ));
            };

            // read constant tensor, add to initializers, and mark as defined in scope
            let tensor = read_tensor(tp, "Constant", out_name)?;
            if !defined.insert(out_name.clone()) {
                return Err(anyhow!(
                    "{node_name}: output '{out_name}' reuses an initializer or graph input name"
                ));
            }
            initializers.push(tensor);
            continue;
        }

        // check supported operator from our runtime
        let Some(op) = Operator::lookup(&np.op_type) else {
            return Err(anyhow!("{node_name}: op '{}' is not supported", np.op_type));
        };
        let spec = op.spec();
        if opset < spec.min_opset {
            return Err(anyhow!(
                "{node_name}: '{}' needs opset ≥ {}, model declares {opset}",
                op.static_str(),
                spec.min_opset
            ));
        }
        if np.input.len() < spec.arity_in.0 || np.input.len() > spec.arity_in.1 {
            return Err(anyhow!(
                "{node_name}: '{}' wants min={}..max={} inputs, got {}",
                op.static_str(),
                spec.arity_in.0,
                spec.arity_in.1,
                np.input.len()
            ));
        }
        if np.output.len() < spec.arity_out.0 || np.output.len() > spec.arity_out.1 {
            return Err(anyhow!(
                "{node_name}: '{}' wants min={}..max={} outputs, got {}",
                op.static_str(),
                spec.arity_out.0,
                spec.arity_out.1,
                np.output.len()
            ));
        }

        // required inputs (first arity_in.0) must be non-empty
        // this check is already done in kernels at runtime, they errors anyway
        for (idx, iname) in np.input.iter().enumerate() {
            if idx < spec.arity_in.0 && iname.is_empty() {
                return Err(anyhow!(
                    "{node_name}: '{}' required input #{} has empty name",
                    op.static_str(),
                    idx
                ));
            }
        }

        // required outputs (first arity_out.0) must be non-empty
        // this is decoding time validation, compute() can peacefully run with strict non-empty outputs
        for (idx, oname) in np.output.iter().enumerate() {
            if idx < spec.arity_out.0 && oname.is_empty() {
                return Err(anyhow!(
                    "{node_name}: '{}' required output #{} has empty name",
                    op.static_str(),
                    idx
                ));
            }
        }

        // validate & collect node attributes
        let mut attrs: Vec<(String, AttributesDataValue)> = Vec::with_capacity(np.attribute.len());
        for a in &np.attribute {
            let Some(&(name, kind)) = spec.attrs.iter().find(|(n, _)| *n == a.name.as_str()) else {
                return Err(anyhow!(
                    "{node_name}: attribute '{}' unknown for '{}'",
                    a.name,
                    op.static_str()
                ));
            };

            // validate attribute type
            let got_type = a.type_;
            let expected_type = kind as i32;
            if a.type_ != Some(expected_type) {
                return Err(anyhow!(
                    "{node_name}: attribute '{name}' expects type {}, got {}",
                    atype(expected_type),
                    if let Some(t) = got_type {
                        atype(t)
                    } else {
                        "none"
                    }
                ));
            }

            // check attr value is present and matches expected kind
            let attr = match (kind, &a.f, &a.i, &a.s, &a.ints) {
                (AttributesDataKind::F32, Some(f), _, _, _) => AttributesDataValue::F32(*f),
                (AttributesDataKind::I64, _, Some(i), _, _) => AttributesDataValue::I64(*i),
                (AttributesDataKind::Str, _, _, Some(s), _) => match String::from_utf8(s.clone()) {
                    Ok(st) => AttributesDataValue::Str(st),
                    Err(_) => {
                        return Err(anyhow!("{node_name}: attribute '{name}' is not valid utf8"));
                    }
                },
                (AttributesDataKind::Ints, _, _, _, _) => AttributesDataValue::Ints(a.ints.clone()),
                _ => {
                    return Err(anyhow!(
                        "{node_name}: attribute '{}' does not carry payload",
                        a.name,
                    ));
                }
            };
            attrs.push((name.to_string(), attr));
        }

        // check for duplicate output names,
        // and that they are not already defined in scope
        for oname in &np.output {
            if oname.is_empty() {
                continue;
            }
            if producers.insert(oname.clone(), i).is_some() {
                return Err(anyhow!("tensor '{oname}': produced by more than one node"));
            }
            if !defined.insert(oname.to_string()) {
                return Err(anyhow!(
                    "{node_name}: output '{oname}' reuses an initializer or graph input name"
                ));
            }
        }

        // collect validate node
        nodes.push(Node {
            op,
            name: node_name,
            inputs: Arc::from(np.input.clone()),
            outputs: Arc::from(np.output.clone()),
            attrs: Arc::from(attrs),
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

    // check that all graph outputs are produced by some node or initializer
    for vo in &graph.output {
        if !vo.name.is_empty() && !defined.contains(vo.name.as_str()) {
            return Err(anyhow!("graph output '{}' is never produced", vo.name));
        }
    }

    // collect operators used in the graph, in topological order, deduped
    let mut dedup_set = HashSet::with_capacity(nodes.len());
    let operators: Vec<String> = nodes
        .iter()
        .filter_map(|node| {
            let op = node.op();
            dedup_set.insert(op).then_some(op)
        })
        .map(|s| s.to_string())
        .collect();

    // init blas-rs thread pool
    blas_rs::init_thread_pool();

    Ok(Model {
        ir_version,
        opset,
        nodes: Arc::from(nodes),
        operators: Arc::from(operators),
        initializers: Arc::from(initializers),
        inputs: Arc::from(inputs),
        outputs: Arc::from(outputs),
    })
}

#[inline]
fn read_tensor(proto: &TensorProto, kind: &str, name: &str) -> Result<Tensor> {
    // only support f32 dtype
    if proto.data_type != 1 {
        return Err(anyhow!(
            "{kind}: '{name}' must be f32 dtype, got {}",
            dtype_name(proto.data_type)
        ));
    }

    // check negative dim & numel overflows
    let mut numel: u64 = 1;
    for (di, &d) in proto.dims.iter().enumerate() {
        if d < 0 {
            return Err(anyhow!(
                "{kind}: '{name}' encountered negative dim value: {d} at index dim[{di}]"
            ));
        }
        numel = numel
            .checked_mul(d as u64)
            .ok_or_else(|| anyhow!("{kind}: '{name}' num of elements overflows"))?;
    }

    // allow empty tensors, for god’s sake,
    // if this creating bad cases to handle, I will lose my shit
    if numel == 0 {
        return Tensor::create(name, &proto.dims, vec![]);
    }

    // collect storage field sources
    let sources: u64 = [
        !proto.raw_data.is_empty(),
        !proto.float_data.is_empty(),
        !proto.int32_data.is_empty(),
        !proto.int64_data.is_empty(),
    ]
    .into_iter()
    .map(|b| b as u64)
    .sum();

    // must have exactly one storage field
    match sources {
        0 => {
            return Err(anyhow!(
                "{kind}: '{name}' {numel} elements, but no data, external storage is not supported"
            ));
        }
        1 => {}
        _ => {
            return Err(anyhow!("{kind}: '{name}' more than one storage field"));
        }
    }

    // must not have int32/int64 data
    if !proto.int32_data.is_empty() || !proto.int64_data.is_empty() {
        return Err(anyhow!(
            "{kind}: '{name}' data in int32/int64 field, only f32 is supported"
        ));
    }

    // create tensor from raw data, if available
    if !proto.raw_data.is_empty() {
        let expected_size = numel
            .checked_mul(4)
            .ok_or_else(|| anyhow!("{kind}: '{name}' byte size overflows"))?;
        if proto.raw_data.len() as u64 != expected_size {
            return Err(anyhow!(
                "{kind}: '{name}' found {} bytes, tensor shape needs {expected_size}",
                proto.raw_data.len()
            ));
        }

        // if !proto.raw_data.len().is_multiple_of(4) {
        //     return Err(anyhow!(
        //         "initializer: raw_data has {} bytes, cannot be converted to f32",
        //         proto.raw_data.len()
        //     ));
        // }

        let (bytes_chunks, _) = proto.raw_data.as_chunks::<4>();
        let mut data = Vec::with_capacity(bytes_chunks.len());
        data.extend(
            bytes_chunks
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );

        Tensor::create(name, &proto.dims, data)
    } else {
        if proto.float_data.len() as u64 != numel {
            return Err(anyhow!(
                "{kind}: '{name}' found {} f32 values, shape needs {numel}",
                proto.float_data.len()
            ));
        }
        let mut data = Vec::with_capacity(proto.float_data.len());
        data.extend(proto.float_data.iter().copied());
        Tensor::create(name, &proto.dims, data)
    }
}

#[inline]
fn read_tensor_spec(proto: &ValueInfoProto, kind: &str) -> Result<TensorSpec> {
    let name = proto.name.as_str();
    let Some(tt) = proto.type_.as_ref().and_then(|t| t.tensor_type.as_ref()) else {
        return Err(anyhow!("{kind} '{name}': no tensor type declared"));
    };
    // proto_code for f32 is 1
    if tt.elem_type != 1 {
        return Err(anyhow!(
            "{kind} '{name}': {}, only f32 is supported",
            dtype_name(tt.elem_type)
        ));
    }
    let Some(shape) = &tt.shape else {
        return Err(anyhow!(
            "{kind} '{name}': no shape declared, concrete shapes required"
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
                    "{kind} '{name}': encountered negative dim value: {v} at index dim[{di}]"
                ));
            }
            None => {
                return Err(anyhow!(
                    "{kind} '{name}': encountered symbolic dim at index dim[{di}], concrete shapes required"
                ));
            }
        }
    }
    Ok(TensorSpec {
        name: Arc::from(name),
        dims: Arc::from(dims),
    })
}

#[inline(always)]
fn dtype_name(proto_code: i32) -> &'static str {
    match proto_code {
        0 => "undefined",
        1 => "float32",
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

#[inline(always)]
fn atype(proto_code: i32) -> &'static str {
    match proto_code {
        1 => "f32",
        2 => "i64",
        3 => "string",
        6 => "floats", // not used, just placeholder
        7 => "ints",
        8 => "strings", // not used, just placeholder
        _ => "unknown",
    }
}
