const NN_Math = require("./neural_network_math");
const Layer = require("./layer");

const Adam = require("./math_functions/optimizers/adam");
const { PropagationMath } = require("./neural_network_math");

class NeuralNetwork {
    constructor() {
        this.layers = [];
        this.loss_function = NN_Math.Losses.BinaryCrossentropy;
        this.optimizer = NN_Math.Optimizers.Adam;
        this.weights = [];
        this.bias = [];
        this.recurrent_weights = [];
        this.recurrent = false;
        this.optimizer_settings = {
            batch_size: 1,
            epochs: 1,
            adam: {
                alpha: 0.001,
                beta1: 0.9,
                beta2: 0.999
            }
        };
    }

    AddLayer(layer) {
        this.layers.push(layer);
        return this;
    }
    SetLoss(loss) {
        this.loss_function = loss;
        return this;
    }
    IsRecurrent() {
        this.recurrent = true;
        return this;
    }
    IsNotRecurrent() {
        this.recurrent = false;
        return this;
    }
    SetOptimizerSettings(settings) {
        this.optimizer_settings = Object.assign(this.optimizer_settings, settings);
        return this;
    }

    Build() {
        for (let i = 0; i < this.layers.length; i++) {
            const not_ouput = i !== this.layers.length - 1;
            const not_input = i !== 0;
            if (not_ouput) {
                this.weights.push((Array(this.layers[i].size * this.layers[i+1].size)).fill().map(() => Math.random()/2-0.25));
            }
            else {
                this.weights.push([]);
            }

            if (not_input) {
                this.bias.push(Array(this.layers[i].size).fill().map(() => Math.random()/2-0.25));
            }
            else {
                this.bias.push([]);
            }

            if (not_input && not_ouput && this.layers[i].recurrent) {
                this.recurrent_weights.push(Array(Math.pow(this.layers[i].size, 2)).fill().map(() => Math.random()/2-0.25));
            }
            else {
                this.recurrent_weights.push([]);
            }
        }
        return this;
    }

    async Predict(inputs) {
        const activation_functions = this.layers.map((layer) => {
            return layer.activation;
        });
        const layer_sizes = this.layers.map((layer) => {
            return layer.size;
        })
        return PropagationMath.ForwardPropagation(
            inputs, this.weights, this.bias, activation_functions, layer_sizes, this.recurrent_weights, true
        );
    }

    async Train(inputs, outputs) {
        const activation_functions = this.layers.map((layer) => {
            return layer.activation;
        });
        const layer_sizes = this.layers.map((layer) => {
            return layer.size;
        })
        switch(this.optimizer) {
            case NN_Math.Optimizers.Adam:
                const weight_updates = new Adam()
                .SetBatchSize(this.optimizer_settings.batch_size)
                .SetEpochs(this.optimizer_settings.epochs)
                .SetAlpha(this.optimizer_settings.adam.alpha)
                .SetBeta1(this.optimizer_settings.adam.beta1)
                .SetBeta2(this.optimizer_settings.adam.beta2)
                .Backpropagation(inputs, outputs, layer_sizes, this.weights, this.bias, activation_functions, this.loss_function, this.recurrent_weights);
                this.weights = weight_updates.weight_gradiants;
                this.bias = weight_updates.bias_gradiants;
                this.recurrent_weights = weight_updates.reccurant_weight_gradiants;
                return this;
        }
    }
}
module.exports = NeuralNetwork;