mod ops;
mod proto;

use crate::proto::ModelProto;
use anyhow::Result;
use core::fmt;
use prost::Message;

#[derive(Debug)]
pub struct ModelValidation {
    pub verdict: Verdict,
    pub failures: Vec<&'static str>,
    pub pending_ops: Vec<&'static str>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Rejected,
    Pending,
    Runnable,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Rejected => write!(f, "Model is beyond the scope of this runtime"),
            Verdict::Pending => write!(f, "Model is valid but some ops are not yet implemented"),
            Verdict::Runnable => write!(f, "Model is valid and runnable"),
        }
    }
}

pub fn decode(bytes: &[u8]) -> Result<ModelValidation> {
    let model = ModelProto::decode(bytes)?;
    let mut failures: Vec<&'static str> = vec![];
    let mut pending_ops: Vec<&'static str> = vec![];

    Ok(ModelValidation {
        verdict,
        failures,
        pending_ops,
    })
}
