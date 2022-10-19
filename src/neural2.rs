use std::{error::Error, io::Write, path::Path};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, EmbeddingConfig, LinearConfig, Module, OptimizerConfig, RNNConfig, VarStore, RNN},
    IndexOp, Kind, Tensor,
};
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NetworkParams {
    /// The learn rate of the network.
    pub learn_rate: f64,
    /// The batch size for the network.
    pub batch_size: usize,
    /// The size of each frame, in samples.
    pub frame_size: usize,
    /// The number of frames to use during training.
    pub num_frames: usize,
    /// The size of the hidden layers.
    pub hidden_size: usize,
    /// The number of RNN layers to use.
    pub rnn_layers: usize,
    /// The size of the embedding
    pub embed_size: usize,
    /// The number of quantization levels to use.
    pub quantization: usize,
    /// Use skip connections in the RNN or not.
    pub skip_connections: bool,
    /// Which epoch the network is currently at
    pub epoch: usize,
}

pub fn print_tensor(tensor: &Tensor, tensor_name: &str, line: u32, epoch: usize) {
    let header = format!(
        "=== {} (line {}, epoch {}): (shape = {:?}) ===",
        tensor_name,
        line,
        epoch,
        tensor.size()
    );
    println!("{}", header);
    tensor.print();
}

pub fn write_csv(file_name: &str, data: &[Vec<f32>]) -> Result<(), Box<dyn Error>> {
    let mut file = std::fs::File::create(file_name)?;
    let data = data
        .iter()
        .map(|x| x.iter().map(|x| x.to_string()).join(","))
        .join(",\n");
    file.write(data.as_bytes())?;
    Ok(())
}

/// Wrapper for the conditioning vector. Has shape [batch_size, num_frames * frame_size, hidden_size]
pub struct ConditioningVector {
    pub tensor: Tensor,
    batch_size: usize,
    num_frames: usize,
    hidden_size: usize,
    frame_size: usize,
}
impl ConditioningVector {
    pub fn new(
        conditioning: Tensor,
        batch_size: usize,
        num_frames: usize,
        hidden_size: usize,
        frame_size: usize,
    ) -> ConditioningVector {
        assert_shape(
            &[batch_size, num_frames * frame_size, hidden_size],
            &conditioning,
        );
        ConditioningVector {
            tensor: conditioning,
            batch_size,
            num_frames,
            hidden_size,
            frame_size,
        }
    }
}

/// A Tensor containing logits, of shape `[batch_size, num_samples, quantization]`
pub struct Logits {
    tensor: Tensor,
    quantization: usize,
    batch_size: usize,
    num_samples: usize,
}

impl Logits {
    pub fn new(
        logits: Tensor,
        batch_size: usize,
        num_samples: usize,
        quantization: usize,
    ) -> Logits {
        assert_shape(&[batch_size * num_samples, quantization], &logits);
        Logits {
            tensor: logits,
            quantization,
            batch_size,
            num_samples,
        }
    }

    pub fn sample(&self) -> Vec<i64> {
        let sample = self
            .tensor
            .softmax(-1, tch::Kind::Float)
            .multinomial(1, false);
        Vec::<i64>::from(sample)
    }

    pub fn sample_one(&self) -> i64 {
        assert_eq!(self.batch_size, 1);
        assert_eq!(self.num_samples, 1);
        let sample = reshape(&[self.quantization], &self.tensor);
        let sample = sample.softmax(-1, tch::Kind::Float).multinomial(1, false);
        assert_shape(&[1], &sample);

        i64::from(sample)
    }
}

pub struct Frames {
    pub tensor: Tensor,
    batch_size: usize,
    num_frames: usize,
    frame_size: usize,
}
impl Frames {
    pub fn new(frames: Tensor, batch_size: usize, num_frames: usize, frame_size: usize) -> Frames {
        assert_shape(&[batch_size, num_frames, frame_size], &frames);
        Frames {
            tensor: frames,
            batch_size,
            num_frames,
            frame_size,
        }
    }

    /// Flatten the tensor into shape `[batch_size * num_frames, frame_size]`
    fn flatten(&self) -> (Tensor, usize) {
        let length = self.batch_size * self.num_frames;
        let tensor = reshape(&[length, self.frame_size], &self.tensor);
        (tensor, length)
    }

    fn from_samples(samples: &[i64]) -> Frames {
        let tensor = Tensor::of_slice(samples);
        let tensor = reshape(&[1, 1, samples.len()], &tensor);
        Frames::new(tensor, 1, 1, samples.len())
    }

    pub fn samples(&self) -> Vec<i64> {
        Vec::<i64>::from(&self.tensor)
    }

    /// Unfolds the frame into windows of size self.frame_size. This will skip the last possible frame
    /// For example, the following: `0 1 2 3 | 4 5 6 7`
    /// is unfolded into:
    /// ```
    /// 0 1 2 3
    ///   1 2 3 4
    ///     2 3 4 5
    ///       3 4 5 6
    /// ```
    /// Notice that `4 5 6 7` is NOT included.
    /// Returned size is equal to `[batch_size, seq_len - frame_size, frame_size]`
    fn unfold(&self) -> Frames {
        // Get a bunch of local sliding windows across the input sequence.
        let frame = reshape(
            &[self.batch_size, self.num_frames * self.frame_size],
            &self.tensor,
        );
        let frame = frame.unfold(1, self.frame_size as i64, 1);

        let unfold_size = self.num_frames * self.frame_size - self.frame_size;
        let frame = frame.narrow(1, 0, unfold_size as i64);

        assert_shape(&[self.batch_size, unfold_size, self.frame_size], &frame);
        Frames::new(frame, self.batch_size, unfold_size, self.frame_size)
    }

    fn seq_len(&self) -> usize {
        self.frame_size * self.num_frames
    }
}

struct LSTMSkipConn {
    lstms: Vec<nn::LSTM>,
    hidden_size: usize,
    frame_size: usize,
}

impl LSTMSkipConn {
    fn new(vs: &VarStore, params: NetworkParams, prefix_name: &str) -> LSTMSkipConn {
        let NetworkParams {
            hidden_size,
            rnn_layers,
            frame_size,
            ..
        } = params;

        let mut lstms = vec![];
        for i in 0..rnn_layers {
            let path = format!("{}_lstm_skip_conn{}", prefix_name, i);

            let in_dim = if i == 0 {
                frame_size
            } else {
                // The size of the layers with a skip connection will be equal to the size of the
                // hidden state tensor concated with the input frame.
                // Hence, it will have size hidden_size + frame_size
                hidden_size + frame_size
            };
            let lstm = lstm(&vs, &path, in_dim, hidden_size, 1);

            lstms.push(lstm)
        }
        LSTMSkipConn {
            lstms,
            hidden_size,
            frame_size,
        }
    }
}

impl RNN for LSTMSkipConn {
    type State = LSTMSkipConnState;

    fn zero_state(&self, batch_dim: i64) -> LSTMSkipConnState {
        let states = self
            .lstms
            .iter()
            .map(|lstm| lstm.zero_state(batch_dim))
            .collect_vec();
        LSTMSkipConnState(states)
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Self::State {
        let input = input.unsqueeze(1);
        let (_output, state) = self.seq_init(&input, in_state);
        state
    }

    fn seq_init(&self, frame: &Tensor, states: &Self::State) -> (Tensor, Self::State) {
        let batch_size = frame.size()[0] as usize;
        let num_frames = frame.size()[1] as usize;
        let frame_size = self.frame_size;
        let hidden_size = self.hidden_size;
        let rnn_layers = self.lstms.len();
        let states = &states.0;

        assert_eq!(self.frame_size as i64, frame.size()[2]);
        assert_eq!(rnn_layers, states.len());

        let mut new_states = Vec::with_capacity(rnn_layers);

        let (mut final_output, new_state) = self.lstms[0].seq_init(&frame, &states[0]);
        assert_shape(&[batch_size, num_frames, hidden_size], &final_output);

        new_states.push(new_state);

        for i in 1..self.lstms.len() {
            assert_shape(&[batch_size, num_frames, hidden_size], &final_output);
            assert_shape(&[batch_size, num_frames, frame_size], &frame);
            let input = Tensor::concat(&[&final_output, &frame], 2);
            assert_shape(&[batch_size, num_frames, hidden_size + frame_size], &input);

            let (output, new_state) = self.lstms[i].seq_init(&input, &states[i]);

            assert_shape(&[batch_size, num_frames, hidden_size], &output);

            final_output = output;
            new_states.push(new_state);
        }

        assert_shape(&[batch_size, num_frames, hidden_size], &final_output);
        (final_output, LSTMSkipConnState(new_states))
    }
}

pub struct LSTMSkipConnState(pub Vec<nn::LSTMState>);

enum LSTMType {
    Normal(nn::LSTM),
    SkipConn(LSTMSkipConn),
}
impl LSTMType {
    fn seq_init(&self, frame: &Tensor, state: &LSTMState) -> (Tensor, LSTMState) {
        match (self, state) {
            (LSTMType::Normal(lstm), LSTMState::Normal(state)) => {
                let (conditioning, state) = lstm.seq_init(&frame, state);
                (conditioning, state.into())
            }
            (LSTMType::SkipConn(lstm), LSTMState::SkipConn(state)) => {
                let (conditioning, state) = lstm.seq_init(&frame, state);
                (conditioning, state.into())
            }
            (LSTMType::Normal(_), LSTMState::SkipConn(_)) => {
                panic!("Expected LSTMState to be Normal, got SkipConn")
            }
            (LSTMType::SkipConn(_), LSTMState::Normal(_)) => {
                panic!("Expected LSTMState to be SkipConn, got Normal")
            }
        }
    }

    fn zero_state(&self, batch_dim: i64) -> LSTMState {
        match self {
            LSTMType::Normal(lstm) => lstm.zero_state(batch_dim).into(),
            LSTMType::SkipConn(lstm) => lstm.zero_state(batch_dim).into(),
        }
    }
}

pub enum LSTMState {
    Normal(nn::LSTMState),
    SkipConn(LSTMSkipConnState),
}

impl From<nn::LSTMState> for LSTMState {
    fn from(value: nn::LSTMState) -> Self {
        LSTMState::Normal(value)
    }
}

impl From<LSTMSkipConnState> for LSTMState {
    fn from(value: LSTMSkipConnState) -> Self {
        LSTMState::SkipConn(value)
    }
}

pub struct FrameLevelRNN {
    lstm: LSTMType,
    linear: nn::Linear,
    hidden_size: usize,
    frame_size: usize,
    quantization: usize,
}

impl FrameLevelRNN {
    fn new(vs: &VarStore, params: NetworkParams) -> FrameLevelRNN {
        let NetworkParams {
            hidden_size,
            frame_size,
            quantization,
            rnn_layers,
            ..
        } = params;

        let lstm = if params.skip_connections {
            let lstm = LSTMSkipConn::new(vs, params, "frame_lstm");
            LSTMType::SkipConn(lstm)
        } else {
            let lstm = lstm(&vs, "frame_lstm", frame_size, hidden_size, rnn_layers);
            LSTMType::Normal(lstm)
        };
        let linear = linear(&vs, "frame_linear", hidden_size, frame_size * hidden_size);
        FrameLevelRNN {
            lstm,
            linear,
            hidden_size,
            frame_size,
            quantization,
        }
    }

    fn forward(&self, frame: &Frames, state: &LSTMState) -> (ConditioningVector, LSTMState) {
        let batch_size = frame.batch_size;
        let num_frames = frame.num_frames;
        let frame_size = self.frame_size;
        let hidden_size = self.hidden_size;
        let quantization = self.quantization;
        let frame = &frame.tensor;
        assert_shape(&[batch_size, num_frames, frame_size], &frame);

        let frame = frame.to_kind(Kind::Float);
        let frame = frame
            .divide_scalar((quantization / 2) as f64)
            .g_sub_scalar(1.0f64)
            .g_mul_scalar(2.0f64);

        let (conditioning, state) = self.lstm.seq_init(&frame, state);
        assert_shape(&[batch_size, num_frames, hidden_size], &conditioning);

        let conditioning = self.linear.forward(&conditioning);
        assert_shape(
            &[batch_size, num_frames, frame_size * hidden_size],
            &conditioning,
        );

        let conditioning = reshape(
            &[batch_size, num_frames * frame_size, hidden_size],
            &conditioning,
        );
        (
            ConditioningVector::new(
                conditioning,
                batch_size,
                num_frames,
                hidden_size,
                frame_size,
            ),
            state,
        )
    }
}

pub struct SamplePredictor {
    embed: nn::Embedding,
    linear_1: nn::Linear,
    linear_2: nn::Linear,
    linear_3: nn::Linear,
    linear_out: nn::Linear,

    quantization: usize,
    frame_size: usize,
    hidden_size: usize,
    embed_size: usize,
}

impl SamplePredictor {
    fn new(vs: &VarStore, params: NetworkParams) -> SamplePredictor {
        let NetworkParams {
            quantization,
            frame_size,
            hidden_size,
            embed_size,
            ..
        } = params;

        let embed = nn::embedding(
            &vs.root() / "samp_pred_embed",
            quantization as i64,
            embed_size as i64,
            EmbeddingConfig::default(),
        );
        let linear_1 = linear(vs, "samp_pred_lin_1", frame_size * embed_size, hidden_size);
        let linear_2 = linear(vs, "samp_pred_lin_2", hidden_size, hidden_size);
        let linear_3 = linear(vs, "samp_pred_lin_3", hidden_size, hidden_size);
        let linear_out = linear(vs, "samp_pred_lin_out", hidden_size, quantization);
        SamplePredictor {
            embed,
            linear_1,
            linear_2,
            linear_3,
            linear_out,
            quantization,
            frame_size,
            hidden_size,
            embed_size,
        }
    }

    fn forward(&self, conditioning: &Tensor, frame: &Frames) -> Logits {
        let embed_size = self.embed_size;
        let hidden_size = self.hidden_size;
        let quantization = self.quantization;
        let frame_size = self.frame_size;
        let batch_size = frame.batch_size;
        let num_frames = frame.num_frames;

        assert_eq!(self.frame_size, frame.frame_size);

        let (frame, length) = frame.flatten();
        assert_eq!(conditioning.size()[0] as usize, length);

        let frame = self.embed.forward(&frame);
        assert_shape(&[length, frame_size, embed_size], &frame);

        let frame = reshape(&[length, frame_size * embed_size], &frame);

        let mut out = self.linear_1.forward(&frame);

        assert_shape(&[length, hidden_size], &out);
        assert_shape(&[length, hidden_size], &conditioning);

        out += conditioning;
        let out = self.linear_2.forward(&out);
        let out = out.relu();
        let out = self.linear_3.forward(&out);
        let out = out.relu();
        let out = self.linear_out.forward(&out);

        assert_shape(&[length, quantization], &out);
        let out = Logits::new(out, batch_size, num_frames, self.quantization);
        out
    }
}

pub struct BackwardsDebug {
    pub loss: f32,
    pub logits: Logits,
    pub targets: Tensor,
}

pub struct NeuralNet {
    frame_level_rnn: FrameLevelRNN,
    sample_predictor: SamplePredictor,
    optim: nn::Optimizer,
    pub params: NetworkParams,
}

impl NeuralNet {
    pub fn new(vs: &VarStore, params: NetworkParams) -> NeuralNet {
        NeuralNet {
            frame_level_rnn: FrameLevelRNN::new(&vs, params),
            sample_predictor: SamplePredictor::new(&vs, params),
            optim: nn::AdamW::default().build(vs, params.learn_rate).unwrap(),
            params: params.clone(),
        }
    }

    pub fn from_saved(
        vs: &mut VarStore,
        model_path: impl AsRef<Path>,
        params_path: impl AsRef<Path>,
    ) -> Result<(NeuralNet, NetworkParams), Box<dyn Error>> {
        let params_str = std::fs::read_to_string(params_path)?;
        let params = serde_json::from_str(&params_str)?;
        let neural_net = NeuralNet::new(vs, params);
        vs.load(model_path)?;
        Ok((neural_net, params))
    }

    pub fn checkpoint(&self, vs: &VarStore, path: &str) -> Result<(), Box<dyn Error>> {
        let params = serde_json::ser::to_string_pretty(&self.params)?;
        vs.save(format!("{}_checkpoint.bin", path))?;
        std::fs::write(format!("{}_checkpoint_params.json", path), params)?;
        Ok(())
    }

    pub fn set_learn_rate(&mut self, learn_rate: f64) {
        self.params.learn_rate = learn_rate;
        self.optim.set_lr(learn_rate);
    }

    pub fn zeros(&self, batch_dim: usize) -> LSTMState {
        self.frame_level_rnn.lstm.zero_state(batch_dim as i64)
    }

    pub fn forward(
        &self,
        mut sliding_window: Vec<i64>,
        state: &LSTMState,
    ) -> (Vec<i64>, LSTMState) {
        let NetworkParams { frame_size, .. } = self.params;

        assert_eq!(sliding_window.len(), frame_size);
        let mut out_samples = Vec::with_capacity(frame_size);

        let mut frame = Frames::from_samples(&sliding_window);
        let (conditioning, state) = self.frame_level_rnn.forward(&frame, state);

        for i in 0..frame_size {
            let conditioning = conditioning
                .tensor
                .i((.., i as i64))
                .reshape(&[-1, self.params.hidden_size as i64]);

            let logits = self.sample_predictor.forward(&conditioning, &frame);
            let sample = logits.sample_one();
            sliding_window.remove(0);
            sliding_window.push(sample);
            assert_eq!(sliding_window.len(), frame_size);

            frame = Frames::from_samples(&sliding_window);
            out_samples.push(sample);
        }

        (out_samples, state)
    }

    pub fn backward(
        &mut self,
        frame: &Frames,
        overlap: &Frames,
        targets: &Frames,
    ) -> BackwardsDebug {
        let NetworkParams {
            batch_size,
            frame_size,
            ..
        } = self.params;
        assert_eq!(frame_size, frame.frame_size);
        assert_eq!(frame_size, overlap.frame_size);
        assert_eq!(frame_size, targets.frame_size);
        assert_eq!(overlap.num_frames, targets.num_frames + 1);

        let zero_state = self.zeros(batch_size);
        let (conditioning, _) = self.frame_level_rnn.forward(&frame, &zero_state);

        let unfolded_overlap = overlap.unfold();
        assert_eq!(unfolded_overlap.num_frames, targets.seq_len());

        // let unfolded_overlap = unfolded_overlap.reshape(&[-1, frame_size as i64]);
        let conditioning = conditioning
            .tensor
            .reshape(&[-1, self.params.hidden_size as i64]);

        let logits = self
            .sample_predictor
            .forward(&conditioning, &unfolded_overlap);
        let targets = reshape(&[batch_size * targets.seq_len()], &targets.tensor);

        let loss = logits.tensor.cross_entropy_for_logits(&targets);

        self.optim.backward_step_clip(&loss, 0.5);

        let loss = f32::from(loss);
        BackwardsDebug {
            loss,
            logits,
            targets,
        }
    }
}

fn linear(vs: &VarStore, path: &str, in_dim: usize, out_dim: usize) -> nn::Linear {
    nn::linear(
        &vs.root() / path,
        in_dim as i64,
        out_dim as i64,
        LinearConfig::default(),
    )
}

fn lstm(
    vs: &VarStore,
    path: &str,
    in_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
) -> nn::LSTM {
    let mut config = RNNConfig::default();
    config.num_layers = num_layers as i64;
    nn::lstm(&vs.root() / path, in_dim as i64, hidden_dim as i64, config)
}

pub fn reshape(shape: &[usize], tensor: &Tensor) -> Tensor {
    let shape = shape.iter().map(|x| *x as i64).collect_vec();
    tensor.reshape(&shape)
}

#[track_caller]
pub fn assert_shape(expected: &[usize], actual: &Tensor) {
    let actual = actual.size();
    let same_len = expected.len() == actual.len();
    let same_values = expected.iter().zip(actual.iter()).all(|(a, b)| {
        let a = *a as i64;
        let b = *b as i64;
        a as i64 == b || (a == 0 && b != 0)
    });
    if !same_len || !same_values {
        panic!(
            "Expected tensor to be of shape {:?}, got {:?}",
            expected, actual
        );
    }
}

#[test]
fn test_unfold() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let expected = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
    ];

    let tensor = Tensor::of_slice(&data);
    let tensor = reshape(&[1, 2, 4], &tensor);
    let frame = Frames::new(tensor, 1, 2, 4);
    let unfolded = frame.unfold();
    assert_eq!(unfolded.num_frames, 5);
    for i in 0..5 {
        let unfolded_frame = tch::IndexOp::i(&unfolded.tensor, (0, i as i64));
        assert_shape(&[4], &unfolded_frame);
        let actual = Vec::<i64>::from(unfolded_frame);
        assert_eq!(actual, expected[i])
    }
}
