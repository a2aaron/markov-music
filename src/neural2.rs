use tch::{
    nn::{self, LSTMState, LinearConfig, Module, OptimizerConfig, RNNConfig, VarStore, RNN},
    Device, Kind, Tensor,
};

use crate::quantize::QuantizedSample;

pub const HIDDEN_SIZE: usize = 1024;
pub const BATCH_SIZE: usize = 1;
pub const SEQ_LEN: usize = 1024;
pub struct NeuralNet {
    lstm: nn::LSTM,
    linear: nn::Linear,
    optim: nn::Optimizer,
    device: Device,
    quantization: usize,
}

impl NeuralNet {
    pub fn new(vs: &VarStore, device: Device, quantization: usize) -> NeuralNet {
        let lstm = lstm(vs, quantization, HIDDEN_SIZE);
        let linear = linear(vs, HIDDEN_SIZE, quantization);

        let optim = nn::AdamW::default().build(vs, 0.01).unwrap();
        NeuralNet {
            lstm,
            device,
            linear,
            optim,
            quantization,
        }
    }

    pub fn zero_state(&self) -> LSTMState {
        self.lstm.zero_state(1)
    }

    pub fn compute(&self, input: i64, state: LSTMState) -> (i64, LSTMState) {
        let input_tensor =
            Tensor::zeros(&[1, self.quantization as i64], (Kind::Float, self.device));
        let _ = input_tensor.narrow(1, input, 1).fill_(1.0);

        let state = self.lstm.step(&input_tensor, &state);
        let output = self.linear.forward(&state.h());
        let output = output
            .squeeze_dim(0)
            .softmax(-1, Kind::Float)
            .multinomial(1, false);
        let output = i64::from(output);
        (output, state)
    }

    pub fn train(&mut self, inputs: &[QuantizedSample]) -> f32 {
        let quantization = self.quantization as i64;
        let seq_len = (SEQ_LEN - 1) as i64;
        let batch_size = BATCH_SIZE as i64;

        let inputs_onehot = Tensor::of_slice(&inputs[0..inputs.len() - 1])
            .view([1, seq_len])
            .onehot(quantization);
        let targets = Tensor::of_slice(&inputs[1..]).view([1, seq_len]);

        let (lstm_out, _) = self.lstm.seq(&inputs_onehot.to_device(self.device));
        let logits = self.linear.forward(&lstm_out);

        let logits = logits.view([batch_size * seq_len, quantization]);
        let targets = targets.view([batch_size * seq_len]);

        let loss = logits.cross_entropy_for_logits(&targets);

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
