# SampleRNN Set Up Instructions

The relevant files for the SampleRNN implementation are `neural2.rs`, containing the network implementation, and `sample_neural.rs`, containing the front-end trainer/generator UI.


# Pre-Requisites

1. Install Rust (nightly)
This repo requires nightly Rust, since it uses some nightly features.

2. Install `tch` dependencies
The SampleRNN implmentation is built using [`tch`](https://github.com/LaurentMazare/tch-rs/). Please follow the installation instructions at that repo (they mostly amount to installing PyTorch).

# How to Use
Quick start: `sample_neural -i path/to/mp3_or_wav_file.mp3 -o path/to/output_folder --length 10 --hidden-size 128 --checkpoint-every 100 --generate-every 100`

Detailed explanation of all flags:
`-i` or `--input` - Takes a path to an MP3 or WAV, and is the training data for the network.
`-o` or `--output` - The path to an output folder. The network will dump a bunch of files there, including generated samples.
`--length` - The length of samples to generate, in seconds.
`--hidden-size` - The size of the hidden layers of the network. I have been using `128` for testing and `1024` when trying to generate something for real
`--checkpoint-every` - How often, in epochs, to generate a checkpoint file
`--generate-every` - How often, in epochs, to generate output samples. Note that this is pretty slow at the moment.

Other flags are available by checking `sample_neural --help`. Most of them have sensible default values and don't need to be messed with.