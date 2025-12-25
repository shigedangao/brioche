use burn::Tensor;
use burn::module::Module;
use burn::nn::Linear;
use burn::prelude::Backend;

#[derive(Module, Debug)]
pub struct VitModule<B: Backend> {
    // Adding a dummy module for the time being.
    linear: Linear<B>,
    pub embeded_dim: usize,
}

impl<B: Backend> VitModule<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        input
    }
}
