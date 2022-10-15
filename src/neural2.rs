use std::io::Write;

use itertools::Itertools;
use tch::{
    nn::{
        self, EmbeddingConfig, LSTMState, LinearConfig, Module, OptimizerConfig, RNNConfig,
        VarStore, RNN,
    },
    Device, Kind, Tensor,
};
thread_local! {
    pub static EPOCH_I: std::cell::RefCell<usize> = std::cell::RefCell::new(0);
}

// Size of batches
pub const BATCH_SIZE: usize = 1;
// Length of BPTT sequence, in frames
pub const NUM_FRAMES: usize = 16;
// Size of frame, in samples
pub const FRAME_SIZE: usize = 15;
// Length of BPTT sequence, in samples (how long is a sequence during backprop)
pub const SEQ_LEN: usize = NUM_FRAMES * FRAME_SIZE;
// Quantization level (256 = 8 bit)
pub const QUANTIZATION: usize = 256;

// Hidden size of the LSTM
// Databots recommended value: 1024?
pub const HIDDEN_SIZE: usize = 256;
// Number of RNNs layers to stack in the LSTM
pub const N_RNN: usize = 1;
// Embedding size (embedding is like one-hot encoding, but
// denser--network learns how to translate QUANTIZATION symbols into EMBED_SIZE dim vector)
// Maybe this can be removed when EMBED_SIZE == QUANTIZATION?
pub const EMBED_SIZE: usize = 256;

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
}
impl ConditioningVector {
    pub fn new(conditioning: Tensor, batch_size: usize, num_frames: usize) -> ConditioningVector {
        assert_shape(
            &[batch_size, num_frames * FRAME_SIZE, HIDDEN_SIZE],
            &conditioning,
        );
        ConditioningVector {
            tensor: conditioning,
            batch_size,
            num_frames,
        }
    }

    // Shorten the conditioning vector along the sample dimension. The output vector has shape
    // [batch_size, shortened_size, HIDDEN_SIZE]
    // Requires 0 < shortened_size && shortened_size <= self.num_frames * FRAME_SIZE
    fn shorten_to(&self, shortened_size: usize) -> Tensor {
        assert!(0 < shortened_size && shortened_size <= self.num_frames * FRAME_SIZE);
        let short = self.tensor.narrow(1, 0, shortened_size as i64);
        assert_shape(&[self.batch_size, shortened_size, HIDDEN_SIZE], &short);
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
}

pub struct FrameLevelRNN {
    lstm: nn::LSTM,
    linear: nn::Linear,
}

impl FrameLevelRNN {
    fn new(vs: &VarStore) -> FrameLevelRNN {
        let lstm = lstm(&vs, FRAME_SIZE, HIDDEN_SIZE, N_RNN);
        let linear = nn::linear(
            &vs.root(),
            HIDDEN_SIZE as i64,
            (FRAME_SIZE * HIDDEN_SIZE) as i64,
            LinearConfig::default(),
        );
        FrameLevelRNN { lstm, linear }
    }

    fn forward(
        &self,
        frame: &Frames,
        state: &LSTMState,
        debug_mode: bool,
    ) -> (ConditioningVector, LSTMState) {
        let frame = &frame.tensor;
        let batch_size = frame.size()[0] as usize;
        let num_frames = frame.size()[1] as usize;
        assert_shape(&[batch_size, num_frames, FRAME_SIZE], &frame);

        let frame = frame.to_kind(Kind::Float);
        let frame = frame
            .divide_scalar((QUANTIZATION / 2) as f64)
            .g_sub_scalar(1.0f64)
            .g_mul_scalar(2.0f64);

        let (conditioning, state) = self.lstm.seq_init(&frame, state);
        assert_shape(&[batch_size, num_frames, HIDDEN_SIZE], &conditioning);

        if debug_mode {
            println!("====== FRAMELEVELRNN::FORWARDS ======");
            debug_tensor(&frame, "frame_float");
            debug_tensor(&conditioning, "conditioning");
        }

        let conditioning = self.linear.forward(&conditioning);
        assert_shape(
            &[batch_size, num_frames, FRAME_SIZE * HIDDEN_SIZE],
            &conditioning,
        );

        let conditioning = reshape(
            &[batch_size, num_frames * FRAME_SIZE, HIDDEN_SIZE],
            &conditioning,
        );
        (
            ConditioningVector::new(conditioning, batch_size, num_frames),
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
}

impl SamplePredictor {
    fn new(vs: &VarStore) -> SamplePredictor {
        let embed = nn::embedding(
            &vs.root(),
            QUANTIZATION as i64,
            EMBED_SIZE as i64,
            EmbeddingConfig::default(),
        );

        // let linear_1 = linear(vs, EMBED_SIZE, HIDDEN_SIZE);
        // I think the pytorch code is wrong--this should take in something of EMBED_SIZE, not FRAME_SIZE * EMBED_SIZE
        // Maybe this was supposed to be RATIO * EMBED_SIZE, where RATIO is the upscale/downscale
        // ratio between tiers?
        let linear_1 = linear(vs, FRAME_SIZE * EMBED_SIZE, HIDDEN_SIZE);

        let linear_2 = linear(vs, HIDDEN_SIZE, HIDDEN_SIZE);
        let linear_3 = linear(vs, HIDDEN_SIZE, HIDDEN_SIZE);
        let linear_out = linear(vs, HIDDEN_SIZE, QUANTIZATION);
        SamplePredictor {
            embed,
            linear_1,
            linear_2,
            linear_3,
            linear_out,
        }
    }

    fn forward(&self, conditioning: &ConditioningVector, frame: &Frames) -> Tensor {
        let batch_size = frame.batch_size;
        let num_frames = frame.num_frames;
        let frame = &frame.tensor;

        // Get a bunch of local sliding windows across the input sequence.
        let frame = reshape(&[batch_size, num_frames * FRAME_SIZE], &frame);
        let frame = frame.unfold(1, FRAME_SIZE as i64, 1);

        let unfold_size = num_frames * FRAME_SIZE - FRAME_SIZE + 1;
        assert_shape(&[batch_size, unfold_size, FRAME_SIZE], &frame);

        let frame = self.embed.forward(&frame);
        assert_shape(&[batch_size, unfold_size, FRAME_SIZE, EMBED_SIZE], &frame);

        let frame = reshape(&[batch_size, unfold_size, FRAME_SIZE * EMBED_SIZE], &frame);

        let mut out = self.linear_1.forward(&frame);
        let conditioning = conditioning.shorten_to(unfold_size);

        assert_shape(&[batch_size, unfold_size, HIDDEN_SIZE], &out);
        assert_shape(&[batch_size, unfold_size, HIDDEN_SIZE], &conditioning);

        out += conditioning;
        let out = self.linear_2.forward(&out);
        let out = out.relu();
        let out = self.linear_3.forward(&out);
        let out = out.relu();
        let out = self.linear_out.forward(&out);

        assert_shape(&[batch_size, unfold_size, QUANTIZATION], &out);

        out
    }
}
pub struct NeuralNet {
    frame_level_rnn: FrameLevelRNN,
    sample_predictor: SamplePredictor,
    optim: nn::Optimizer,
    device: Device,
}

impl NeuralNet {
    pub fn new(vs: &VarStore, device: Device) -> NeuralNet {
        let optim = nn::AdamW::default().build(vs, 0.01).unwrap();
        NeuralNet {
            frame_level_rnn: FrameLevelRNN::new(&vs),
            sample_predictor: SamplePredictor::new(&vs),
            device,
            optim,
        }
    }

    pub fn zeros(&self, batch_dim: usize) -> (Tensor, LSTMState) {
        let frame = Tensor::zeros(
            &[batch_dim as i64, 1, FRAME_SIZE as i64],
            (Kind::Int64, self.device),
        );
        let state = self.frame_level_rnn.lstm.zero_state(batch_dim as i64);
        (frame, state)
    }

    pub fn forward(
        &self,
        frame: &Frames,
        state: &LSTMState,
        debug_mode: bool,
    ) -> (Frames, LSTMState) {
        let (conditioning, state) = self.frame_level_rnn.forward(frame, state, debug_mode);
        let logits = self.sample_predictor.forward(&conditioning, &frame);
        let samples = logits
            .squeeze_dim(0)
            .softmax(-1, Kind::Float)
            .multinomial(1, false);
        if debug_mode {
            println!("====== NEURALNET::FORWARDS ======");
            debug_tensor(&frame.tensor, "frame_int");
            let samples = reshape(&[1, FRAME_SIZE], &samples);
            debug_tensor(&samples, "samples_out");
        }

        let samples = reshape(&[frame.batch_size, 1, frame.frame_size], &samples);
        (
            Frames::new(samples, frame.batch_size, 1, frame.frame_size),
            state,
        )
    }

    pub fn backward(&mut self, frame: &Frames, targets: &Frames, debug_mode: bool) -> f32 {
        let zero_state = self.zeros(BATCH_SIZE).1;
        let (conditioning, _) = self
            .frame_level_rnn
            .forward(&frame, &zero_state, debug_mode);

        let logits = self.sample_predictor.forward(&conditioning, &frame);

        let unfold_size = frame.num_frames * FRAME_SIZE - FRAME_SIZE + 1;

        assert_shape(&[BATCH_SIZE, unfold_size, QUANTIZATION], &logits);

        let batch_size = BATCH_SIZE as i64;
        let seq_len = SEQ_LEN as i64;
        let quantization = QUANTIZATION as i64;

        let logits_view = logits.view([batch_size * unfold_size as i64, quantization]);

        let targets = targets.tensor.view([batch_size, seq_len]);
        let targets = targets.narrow(1, 0, unfold_size as i64);
        let targets = targets.view([batch_size * unfold_size as i64]);

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
