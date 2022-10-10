use itertools::Itertools;
use rand::Rng;

#[derive(Debug)]
pub struct FullNetwork {
    pub layers: Vec<Layer>,
}

/// A full neural network with some number of layers.
/// Note that many functions here are &mut self instead of &self because nodes
/// will store their activations and weighted inputs due to backprop needing them.
/// This might change in the future if I can figure out a less silly way to do this.
impl FullNetwork {
    /// Creates a full network, with the number of specified inputs on the first
    /// layer, and the sizes for each layer. Note that the last layer should
    /// be the output layer, and therefore have as many nodes as your output.
    /// The ith layer should have `layer_sizes[i]` nodes, and each node in that layer
    /// will have `layer_sizes[i - 1]` weights going to it, or `num_inputs_first_layer`
    /// weights if the node is on the first layer.
    pub fn with_size(num_inputs_first_layer: usize, layer_sizes: &[usize]) -> FullNetwork {
        let mut layers = Vec::with_capacity(layer_sizes.len());
        // The first layer connects to the inputs, so each node in it must
        // connect to `num_inputs_first_layer` inputs, and there will be
        // `layer_sizes[0]` nodes in total.
        let first_layer = Layer::with_sizes(num_inputs_first_layer, layer_sizes[0]);
        layers.push(first_layer);
        // Create the rest of the layers.
        // Note that here we wish for `size` to range over 1..layer_sizes.len()
        // and for `prev_layer_size` to range over 0..layer_sizes.len() - 1
        // The - 1 is implicit as `zip` will stop at the first iterator that runs out
        // which happens to be layer_sizes[1..].
        for (&size, &prev_layer_size) in layer_sizes[1..].iter().zip(layer_sizes) {
            // The number of inputs of a layer is equal to the size of the previous layer, since each node in the prior layer is sending
            // their output to each node of the current layer, hence each current
            // layer node recieves `prev_layer_size` inputs. There will be `size` nodes
            // in the current layer in all.
            let layer = Layer::with_sizes(prev_layer_size, size);
            layers.push(layer);
        }
        FullNetwork { layers }
    }

    /// Compute the inputs through the whole network, returning the output
    /// of the last layer.
    pub fn compute(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut output = self.layers[0].compute(inputs);
        for layer in &mut self.layers[1..] {
            output = layer.compute(&output);
        }
        output
    }

    /// Compute the cost, which is the squared distance between the
    /// expected ouput and the actual output of the network. `expected_output`
    /// must be the same size as the vector returned by `self.compute(inputs)` or
    /// this function will panic.
    pub fn cost(&mut self, inputs: &[f32], expected_output: &[f32]) -> f32 {
        let output = self.compute(inputs);
        distance2(&output, &expected_output)
    }

    // eta = learning rate
    pub fn backprop(&mut self, eta: f32, inputs: &[f32], expected_output: &[f32]) {
        // First, ask the network what it thinks the answer is.
        // This also makes all the nodes compute their weighted_input
        let result = self.compute(inputs);
        // println!("Initial result: {:?}", result);

        // Now calculate the output error of the last layer
        let last_layer = self.layers.last().unwrap();
        let al_minus_y = last_layer
            .activation()
            .zip_eq(expected_output)
            .map(|(a, y)| a - y);
        let sigmoid_prime_z = last_layer.weighted_input().map(|z| sigmoid_prime(z));

        let mut output_error: Vec<_> = al_minus_y
            .zip_eq(sigmoid_prime_z)
            .map(|(a, b)| a * b)
            .collect();
        // println!("Last layer output error: {:?}", output_error);

        for i in (0..self.layers.len() - 1).rev() {
            // Split layers here because we wish to have two mutable borrows to self.layers
            // (which is normally disallowed), but we know that these borrows do not overlap
            // because we want two different elements. Splitting it lets us borrow as desired
            let (before, after) = self.layers.split_at_mut(i + 1);

            let this_layer = &mut before.last().unwrap();
            let forward_layer = &mut after[0];

            // compute the error for the previous layer using the forward layer
            let forward_weights: Vec<_> =
                forward_layer.weights_times_error(&output_error).collect();
            let sigmoid_prime_z: Vec<_> = this_layer
                .weighted_input()
                .map(|z| sigmoid_prime(z))
                .collect();

            let this_output_error = forward_weights
                .iter()
                .zip_eq(sigmoid_prime_z)
                .map(|(a, b)| a * b)
                .collect();
            // println!("Layer {} output error: {:?}", i, this_output_error);

            // update the weights in the forward layer
            for (j, node) in forward_layer.nodes.iter_mut().enumerate() {
                for (k, weight) in node.weights.iter_mut().enumerate() {
                    *weight -= eta * output_error[j] * this_layer.nodes[k].activation;
                }
                node.bias -= eta * output_error[j];
            }

            output_error = this_output_error;
        }

        for (j, node) in self.layers[0].nodes.iter_mut().enumerate() {
            for (k, weight) in node.weights.iter_mut().enumerate() {
                *weight -= eta * output_error[j] * inputs[k];
            }
            node.bias -= eta * output_error[j];
        }
    }

    pub fn print_weights(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            for (j, node) in layer.nodes.iter().enumerate() {
                for (k, weight) in node.weights.iter().enumerate() {
                    print!("w{}{}{} = {};", i, j, k, weight);
                }
                print!("b{}{} = {};", i, j, node.bias)
            }
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    pub nodes: Vec<Node>,
}

impl Layer {
    pub fn with_sizes(num_inputs: usize, num_nodes: usize) -> Layer {
        Layer {
            nodes: (0..num_nodes)
                .map(|_| Node::with_size(num_inputs))
                .collect(),
        }
    }

    pub fn compute(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.nodes
            .iter_mut()
            .map(|node| node.compute(inputs))
            .collect()
    }

    pub fn weights_times_error<'a>(&'a self, error: &'a [f32]) -> impl Iterator<Item = f32> + 'a {
        debug_assert!(error.len() == self.nodes.len());
        // iterate by the number of weights for each node
        (0..self.nodes[0].weights.len()).map(move |i| {
            self.nodes
                .iter()
                .zip_eq(error)
                .map(move |(n, e)| n.weights[i] * e)
                .sum()
        })
    }

    pub fn weighted_input<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.nodes.iter().map(|node| node.weighted_input)
    }

    pub fn activation<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.nodes.iter().map(|node| node.activation)
    }
}

#[derive(Debug)]
pub struct Node {
    // The INCOMING WEIGHT (basically, this weight is connected to the previous)
    // layer, not the next.
    pub weights: Vec<f32>,
    pub bias: f32,
    // Equal to weights * a + bias
    // where a is the input to the node
    pub weighted_input: f32,
    // Equal to sigmoid(weighted_input)
    pub activation: f32,
}

impl Node {
    pub fn with_size(num_inputs: usize) -> Node {
        let mut rng = rand::thread_rng();
        Node {
            weights: (0..num_inputs).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            bias: rng.gen_range(-1.0..1.0),
            weighted_input: 0.0,
            activation: 0.0,
        }
    }

    pub fn compute(&mut self, inputs: &[f32]) -> f32 {
        debug_assert!(inputs.len() == self.weights.len());
        let dot_product: f32 = dot(&self.weights, inputs);
        self.weighted_input = dot_product + self.bias;
        self.activation = sigmoid(dot_product + self.bias);
        self.activation
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip_eq(b).map(|(a, b)| a * b).sum()
}

fn distance2(a: &[f32], b: &[f32]) -> f32 {
    // || a - b || ^ 2
    a.iter().zip_eq(b).map(|(a, b)| (a - b) * (a - b)).sum()
}

fn sigmoid(input: f32) -> f32 {
    1.0 / (1.0 + (-input).exp())
}

fn sigmoid_prime(input: f32) -> f32 {
    sigmoid(input) * (1.0 - sigmoid(input))
}
