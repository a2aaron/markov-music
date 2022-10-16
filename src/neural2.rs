use std::io::Write;

use clap::Parser;
use itertools::Itertools;
use tch::{
    nn::{
        self, EmbeddingConfig, LSTMState, LinearConfig, Module, OptimizerConfig, RNNConfig,
        VarStore, RNN,
    },
    Kind, Tensor,
};
thread_local! {
    pub static EPOCH_I: std::cell::RefCell<usize> = std::cell::RefCell::new(0);
}

#[derive(Parser, Debug, Clone, Copy)]
pub struct NetworkParams {
    /// The learn rate of the network.
    #[arg(long, default_value_t = 0.001)]
    pub learn_rate: f64,
    /// The batch size for the network.
    #[arg(long, default_value_t = 128)]
    pub batch_size: usize,
    /// The size of each frame, in samples.
    #[arg(long, default_value_t = 16)]
    pub frame_size: usize,
    /// The number of frames to use during training.
    #[arg(long, default_value_t = 64)]
    pub num_frames: usize,
    /// The size of the hidden layers.
    #[arg(long, default_value_t = 1024)]
    pub hidden_size: usize,
    /// The number of RNN layers to use.
    #[arg(long, default_value_t = 5)]
    pub rnn_layers: usize,
    /// The size of the embedding
    #[arg(long, default_value_t = 256)]
    pub embed_size: usize,
    /// The number of quantization levels to use.
    #[arg(long, default_value_t = 256)]
    pub quantization: usize,
}

fn debug_tensor(tensor: &Tensor, name: &str) {
    let epoch = EPOCH_I.with(|i| *i.borrow());
    // print_tensor(&$var, name, line, epoch);
    let file_name = format!("outputs/{}_epoch_{}.csv", name, epoch);
    write_tensor(&file_name, tensor);
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

fn write_tensor(file_name: &str, data: &Tensor) {
    let shape: Vec<f32> = data.size().into_iter().map(|x| x as f32).collect();
    let data: Vec<f32> = data.into();
    write_csv(file_name, &[shape, data]);
}

pub fn write_csv(file_name: &str, data: &[Vec<f32>]) {
    let mut file = std::fs::File::create(file_name).unwrap();
    let data = data
        .iter()
        .map(|x| x.iter().map(|x| x.to_string()).join(","))
        .join(",\n");
    file.write(data.as_bytes()).unwrap();
}

/// Wrapper for the conditioning vector. Has shape [batch_size, num_frames * FRAME_SIZE, HIDDEN_SIZE]
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

    // Shorten the conditioning vector along the sample dimension. The output vector has shape
    // [batch_size, shortened_size, hidden_size]
    // Requires 0 < shortened_size && shortened_size <= self.num_frames * frame_size
    fn shorten_to(&self, shortened_size: usize) -> Tensor {
        assert!(0 < shortened_size && shortened_size <= self.num_frames * self.frame_size);
        let short = self.tensor.narrow(1, 0, shortened_size as i64);
        assert_shape(&[self.batch_size, shortened_size, self.hidden_size], &short);
        short
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

    fn from_samples(samples: &[i64]) -> Frames {
        let tensor = Tensor::of_slice(samples);
        let tensor = reshape(&[1, 1, samples.len()], &tensor);
        Frames::new(tensor, 1, 1, samples.len())
    }

    pub fn samples(&self) -> Vec<i64> {
        Vec::<i64>::from(&self.tensor)
    }

    fn unfold(&self) -> (Tensor, usize) {
        // Get a bunch of local sliding windows across the input sequence.
        let frame = reshape(
            &[self.batch_size, self.num_frames * self.frame_size],
            &self.tensor,
        );
        let frame = frame.unfold(1, self.frame_size as i64, 1);

        let unfold_size = self.num_frames * self.frame_size - self.frame_size + 1;
        assert_shape(&[self.batch_size, unfold_size, self.frame_size], &frame);
        (frame, unfold_size)
    }
}

pub struct FrameLevelRNN {
    lstm: nn::LSTM,
    linear: nn::Linear,
    hidden_size: usize,
    frame_size: usize,
    quantization: usize,
}

impl FrameLevelRNN {
    fn new(vs: &VarStore, params: NetworkParams) -> FrameLevelRNN {
        let NetworkParams {
            hidden_size,
            rnn_layers,
            frame_size,
            quantization,
            ..
        } = params;

        let lstm = lstm(&vs, frame_size, hidden_size, rnn_layers);
        let linear = nn::linear(
            &vs.root(),
            hidden_size as i64,
            (frame_size * hidden_size) as i64,
            LinearConfig::default(),
        );
        FrameLevelRNN {
            lstm,
            linear,
            hidden_size,
            frame_size,
            quantization,
        }
    }

    fn forward(
        &self,
        frame: &Frames,
        state: &LSTMState,
        debug_mode: bool,
    ) -> (ConditioningVector, LSTMState) {
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

        if debug_mode {
            println!("====== FRAMELEVELRNN::FORWARDS ======");
            debug_tensor(&frame, "frame_float");
            debug_tensor(&conditioning, "conditioning");
        }

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
            &vs.root(),
            quantization as i64,
            embed_size as i64,
            EmbeddingConfig::default(),
        );
        let linear_1 = linear(vs, frame_size * embed_size, hidden_size);
        let linear_2 = linear(vs, hidden_size, hidden_size);
        let linear_3 = linear(vs, hidden_size, hidden_size);
        let linear_out = linear(vs, hidden_size, quantization);
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

    fn forward(&self, conditioning: &ConditioningVector, frame: &Tensor) -> Tensor {
        let batch_size = frame.size()[0] as usize;
        let unfold_size = frame.size()[1] as usize;
        let frame_size = self.frame_size;
        let embed_size = self.embed_size;
        let hidden_size = self.hidden_size;
        let quantization = self.quantization;

        let frame = &frame;

        let frame = self.embed.forward(&frame);
        assert_shape(&[batch_size, unfold_size, frame_size, embed_size], &frame);

        let frame = reshape(&[batch_size, unfold_size, frame_size * embed_size], &frame);

        let mut out = self.linear_1.forward(&frame);
        let conditioning = conditioning.shorten_to(unfold_size);

        assert_shape(&[batch_size, unfold_size, hidden_size], &out);
        assert_shape(&[batch_size, unfold_size, hidden_size], &conditioning);

        out += conditioning;
        let out = self.linear_2.forward(&out);
        let out = out.relu();
        let out = self.linear_3.forward(&out);
        let out = out.relu();
        let out = self.linear_out.forward(&out);

        assert_shape(&[batch_size, unfold_size, quantization], &out);

        out
    }
}
pub struct NeuralNet {
    frame_level_rnn: FrameLevelRNN,
    sample_predictor: SamplePredictor,
    optim: nn::Optimizer,
    pub params: NetworkParams,
}

impl NeuralNet {
    pub fn new(vs: &VarStore, params: NetworkParams) -> NeuralNet {
        let optim = nn::AdamW::default().build(vs, params.learn_rate).unwrap();
        NeuralNet {
            frame_level_rnn: FrameLevelRNN::new(&vs, params),
            sample_predictor: SamplePredictor::new(&vs, params),
            optim,
            params: params.clone(),
        }
    }

    pub fn zeros(&self, batch_dim: usize) -> LSTMState {
        self.frame_level_rnn.lstm.zero_state(batch_dim as i64)
    }

    pub fn forward(
        &self,
        mut sliding_window: Vec<i64>,
        state: &LSTMState,
        debug_mode: bool,
    ) -> (Vec<i64>, LSTMState) {
        let NetworkParams {
            frame_size,
            quantization,
            ..
        } = self.params;

        assert_eq!(sliding_window.len(), frame_size);
        let mut out_samples = Vec::with_capacity(frame_size);

        let mut frame = Frames::from_samples(&sliding_window);
        let (conditioning, state) = self.frame_level_rnn.forward(&frame, state, debug_mode);

        for _ in 0..frame_size {
            let logits = self.sample_predictor.forward(&conditioning, &frame.tensor);

            assert_shape(&[1, 1, quantization], &logits);
            let sample = reshape(&[quantization], &logits);
            let sample = sample.softmax(-1, Kind::Float).multinomial(1, false);
            assert_shape(&[1], &sample);
            let sample = i64::from(sample);

            sliding_window.remove(0);
            sliding_window.push(sample);
            assert_eq!(sliding_window.len(), frame_size);

            frame = Frames::from_samples(&sliding_window);
            out_samples.push(sample);
        }

        (out_samples, state)
    }

    pub fn backward(&mut self, frame: &Frames, targets: &Frames, debug_mode: bool) -> f32 {
        let NetworkParams {
            batch_size,
            frame_size,
            num_frames,
            quantization,
            ..
        } = self.params;

        let zero_state = self.zeros(batch_size);
        let (conditioning, _) = self
            .frame_level_rnn
            .forward(&frame, &zero_state, debug_mode);

        let (unfolded_frame, unfold_size) = frame.unfold();
        let logits = self
            .sample_predictor
            .forward(&conditioning, &unfolded_frame);

        assert_shape(&[batch_size, unfold_size, quantization], &logits);

        let seq_len = num_frames * frame_size;

        let logits_view = reshape(&[batch_size * unfold_size, quantization], &logits);

        let targets = reshape(&[batch_size, seq_len], &targets.tensor);
        let targets = targets.narrow(1, 0, unfold_size as i64);
        let targets = reshape(&[batch_size * unfold_size], &targets);

        if debug_mode {
            println!("====== NEURALNET::BACKWARDS ======");
            debug_tensor(&frame.tensor, "frame_back");
            debug_tensor(&targets, "targets_back");
            debug_tensor(&logits, "logits_back");
        }

        let loss = logits_view.cross_entropy_for_logits(&targets);

        self.optim.backward_step_clip(&loss, 0.5);

        EPOCH_I.with(|i| *i.borrow_mut() += 1);
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

fn lstm(vs: &VarStore, in_dim: usize, hidden_dim: usize, num_layers: usize) -> nn::LSTM {
    let mut config = RNNConfig::default();
    config.num_layers = num_layers as i64;
    nn::lstm(&vs.root(), in_dim as i64, hidden_dim as i64, config)
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
