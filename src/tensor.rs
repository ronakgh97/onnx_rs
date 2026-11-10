//! Tensor utilities
use anyhow::{Result, anyhow};
use std::sync::Arc;

/// Tensor of f32 values with name and shape
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: Arc<str>,
    pub dims: Arc<[i64]>,
    pub data: Arc<[f32]>,
}

/// Tensor specification with name and shape
#[derive(Clone, Debug, PartialEq)]
pub struct TensorSpec {
    pub name: Arc<str>,
    pub dims: Arc<[i64]>,
}

impl Tensor {
    pub fn create(name: &str, dims: &[i64], data: &[f32]) -> Result<Self> {
        // calculate numel with overflow checks
        let numel = dims
            .iter()
            .try_fold(1usize, |a, &d| {
                usize::try_from(d).ok().and_then(|d| a.checked_mul(d))
            })
            .ok_or_else(|| {
                anyhow::anyhow!("Tensor '{name}': dims {dims:?} overflow element count")
            })?;
        if numel != data.len() {
            return Err(anyhow::anyhow!(
                "Tensor '{name}': dims {dims:?} has {numel} elements, but data has {} elements",
                data.len()
            ));
        }
        Ok(Self {
            name: Arc::from(name),
            dims: Arc::from(dims),
            data: Arc::from(data),
        })
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn strides(&self) -> Vec<i64> {
        let mut strides = vec![1i64; self.dims.len()];
        if !self.dims.is_empty() {
            for i in (0..self.dims.len() - 1).rev() {
                strides[i] = strides[i + 1] * self.dims[i + 1];
            }
        }
        strides
    }
}

#[inline(always)]
pub fn take_tensor<'s>(
    inputs: &'s [Option<&'s Tensor>],
    index: usize,
    op_name: &str,
) -> Result<&'s Tensor> {
    let Some(t) = inputs.get(index).copied().flatten() else {
        return Err(anyhow!("{op_name}: required input slot #{index} is absent"));
    };
    Ok(t)
}
