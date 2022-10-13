use tch::{
    nn::{self, LSTMState, LinearConfig, Module, OptimizerConfig, RNNConfig, VarStore, RNN},
    Device, Reduction, Tensor,
};

pub const HIDDEN_SIZE: usize = 256;
pub const BATCH_SIZE: usize = 1;
pub const SEQ_LEN: usize = 128;
pub struct NeuralNet {
    lstm: nn::LSTM,
    linear: nn::Linear,
    optim: nn::Optimizer,
    device: Device,
}

impl NeuralNet {
    pub fn new(vs: &VarStore, device: Device) -> NeuralNet {
        let lstm = lstm(vs, 1, HIDDEN_SIZE);
        let linear = linear(vs, HIDDEN_SIZE, 1);

        let optim = nn::AdamW::default().build(vs, 0.01).unwrap();
        NeuralNet {
            lstm,
            device,
            linear,
            optim,
        }
    }

    pub fn zero_state(&self) -> LSTMState {
        self.lstm.zero_state(1)
    }

    pub fn compute(&self, input: f32, state: LSTMState) -> (f32, LSTMState) {
        let input_tensor = Tensor::of_slice(&[input]);
        let state = self.lstm.step(&input_tensor, &state);
        let output = self.linear.forward(&state.h());
        let output = f32::from(output);
        (output, state)
    }

    pub fn train(&mut self, input: [f32; SEQ_LEN]) -> f32 {
        let input_tensor =
            Tensor::of_slice(&input[..input.len() - 1]).view([1, (SEQ_LEN - 1) as i64, 1]);

        let target = Tensor::of_slice(&input[1..]).view([1, (SEQ_LEN - 1) as i64, 1]);

        let (lstm_out, _) = self.lstm.seq(&input_tensor.to_device(self.device));
        let output = self.linear.forward(&lstm_out);
        // println!("{:?} {:?}", input_tensor.size(), output.size());

        let loss = output.mse_loss(&target, Reduction::Sum);
        self.optim.backward_step_clip(&loss, 0.5);
        f32::from(loss)
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

fn lstm(vs: &VarStore, in_dim: usize, hidden_dim: usize) -> nn::LSTM {
    nn::lstm(
        &vs.root(),
        in_dim as i64,
        hidden_dim as i64,
        RNNConfig::default(),
    )
}
