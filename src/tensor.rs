use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dims: Arc<[i64]>,
    pub data: Arc<[f32]>,
}

impl Tensor {
    pub fn create(name: String, dims: Vec<i64>, data: Vec<f32>) -> Self {
        assert_eq!(
            dims.iter().product::<i64>() as usize,
            data.len(),
            "Tensor dimensions do not match data length"
        );
        Self {
            name,
            dims: Arc::from(dims),
            data: Arc::from(data),
        }
    }
    pub fn len(&self) -> usize {
        self.dims.iter().product::<i64>() as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn shape(&self) -> &[i64] {
        &self.dims
    }
}
