use tch::{
    nn::{self, LinearConfig, Module, OptimizerConfig, VarStore},
    Tensor,
};

pub const IN_WINDOW_SIZE: usize = 128;
pub const OUT_WINDOW_SIZE: usize = 128;

pub struct NeuralNet {
    model: Box<dyn Module>,
    optim: nn::Optimizer,
}

impl NeuralNet {
    pub fn new(vs: &VarStore) -> NeuralNet {
        let model = nn::seq()
            .add(linear(vs, IN_WINDOW_SIZE, 128))
            .add_fn(|xs| xs.relu())
            .add(linear(vs, 128, 128))
            .add_fn(|xs| xs.relu())
            .add(linear(vs, 128, OUT_WINDOW_SIZE));
        println!("{:?}", model);
        let optim = nn::AdamW::default().build(vs, 0.01).unwrap();
        NeuralNet {
            model: Box::new(model),
            optim,
        }
    }

    pub fn compute(&self, input: [f32; IN_WINDOW_SIZE]) -> Vec<f32> {
        let input = Tensor::of_slice(&input);
        let pred = self.model.forward(&input);
        pred.into()
    }

    pub fn train(&mut self, input: [f32; IN_WINDOW_SIZE], output: [f32; OUT_WINDOW_SIZE]) -> f32 {
        let input = Tensor::of_slice(&input);
        let targ = Tensor::of_slice(&output);

        let pred = self.model.forward(&input);

        let loss = pred.mse_loss(&targ, tch::Reduction::Mean);
        let loss_value = loss.data().into();

        self.optim.backward_step_clip(&loss, 0.5);
        loss_value
    }
}

fn linear(vs: &VarStore, in_dim: usize, out_dim: usize) -> nn::Linear {
    nn::linear(
        &vs.root(),
        in_dim as i64,
        out_dim as i64,
        LinearConfig::default(),
    )
}
