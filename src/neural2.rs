use dfdx::{
    prelude::{mse_loss, Linear, Module, Momentum, Optimizer, ResetParams, Sgd, SgdConfig, Tanh},
    tensor::{HasArrayData, Tensor1D, TensorCreator},
};

type Model = (Linear<2, 3>, Tanh, Linear<3, 3>, Linear<3, 1>);

pub struct NeuralNet {
    model: Model,
    sgd: Sgd<Model>,
}

impl NeuralNet {
    pub fn new() -> NeuralNet {
        let mut model = Model::default();
        model.reset_params(&mut rand::thread_rng());

        let sgd = Sgd::new(SgdConfig {
            lr: 1e-1,
            momentum: Some(Momentum::Nesterov(0.5)),
        });
        NeuralNet { model, sgd }
    }

    pub fn reset(&mut self) {
        self.model.reset_params(&mut rand::thread_rng());
        self.sgd = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: None,
        });
    }

    pub fn compute(&self, input: [f32; 2]) -> f32 {
        let input = Tensor1D::new(input);
        let pred = self.model.forward(input);
        pred.data()[0]
    }

    pub fn compute_with_loss(&self, input: [f32; 2], output: f32) -> (f32, f32) {
        let input = Tensor1D::new(input);
        let targ = Tensor1D::new([output]);
        let pred = self.model.forward(input);
        let out = pred.data()[0];

        let loss = mse_loss(pred, &targ);
        (out, *loss.data())
    }

    pub fn train(&mut self, input: [f32; 2], output: f32) {
        let input = Tensor1D::new(input).trace();
        let targ = Tensor1D::new([output]);

        let pred = self.model.forward_mut(input);

        let loss = mse_loss(pred, &targ);
        let gradients = loss.backward();

        self.sgd
            .update(&mut self.model, gradients)
            .expect("Oops, there were some unused params");
    }
}
