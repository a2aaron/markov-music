use dfdx::{
    prelude::{mse_loss, Adam, AdamConfig, Linear, Module, Optimizer, ResetParams, Tanh},
    tensor::{HasArrayData, Tensor1D, TensorCreator},
};

pub const IN_WINDOW_SIZE: usize = 128;
pub const OUT_WINDOW_SIZE: usize = 128;
type Model = (
    (Linear<IN_WINDOW_SIZE, IN_WINDOW_SIZE>, Tanh),
    (Linear<IN_WINDOW_SIZE, 512>, Tanh),
    (Linear<512, 256>, Tanh),
    (Linear<256, 512>, Tanh),
    (Linear<512, OUT_WINDOW_SIZE>, Tanh),
);

pub struct NeuralNet {
    model: Model,
    optim: Adam<Model>,
}

impl NeuralNet {
    pub fn new() -> NeuralNet {
        let mut model = Model::default();
        model.reset_params(&mut rand::thread_rng());

        let optim = Adam::new(AdamConfig {
            lr: 1e-5,
            betas: [0.9, 0.999],
            eps: 1e-8,
        });
        NeuralNet { model, optim }
    }

    pub fn compute(&self, input: [f32; IN_WINDOW_SIZE]) -> Vec<f32> {
        let input = Tensor1D::new(input);
        let pred = self.model.forward(input);
        pred.data().to_vec()
    }

    pub fn train(&mut self, input: [f32; IN_WINDOW_SIZE], output: [f32; OUT_WINDOW_SIZE]) -> f32 {
        let input = Tensor1D::new(input).trace();
        let targ = Tensor1D::new(output);

        let pred = self.model.forward_mut(input);

        let loss = mse_loss(pred, &targ);
        let loss_value = *loss.data();

        let gradients = loss.backward();

        self.optim
            .update(&mut self.model, gradients)
            .expect("Oops, there were some unused params");
        loss_value
    }
}
