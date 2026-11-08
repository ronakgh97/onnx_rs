//! Tensor utilities
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: Arc<str>,
    pub dims: Arc<[i64]>,
    pub data: Arc<[f32]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorSpec {
    pub name: Arc<str>,
    pub dims: Arc<[i64]>,
}

impl Tensor {
    pub fn create(name: String, dims: Vec<i64>, data: Vec<f32>) -> Self {
        assert_eq!(
            dims.iter().product::<i64>() as usize,
            data.len(),
            "Tensor dimensions do not match data length"
        );
        Self {
            name: Arc::from(name),
            dims: Arc::from(dims),
            data: Arc::from(data),
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product::<i64>() as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn strides(&self) -> Vec<i64> {
        let mut strides = vec![1i64; self.dims.len()];
        for i in (0..self.dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }
}
