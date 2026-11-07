//! Minimal protobuf definitions for parsing .onnx files
use prost::Message;

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, optional, tag = "1")]
    pub ir_version: Option<i64>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    pub domain: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, optional, tag = "2")]
    pub f: Option<f32>,
    #[prost(int64, optional, tag = "3")]
    pub i: Option<i64>,
    #[prost(bytes = "vec", optional, tag = "4")]
    pub s: Option<Vec<u8>>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(int32, optional, tag = "20")]
    pub type_: Option<i32>,
}

#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub type_: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TypeProtoTensor>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProtoTensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<DimProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct DimProto {
    #[prost(int64, optional, tag = "1")]
    pub dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub dim_param: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
}
