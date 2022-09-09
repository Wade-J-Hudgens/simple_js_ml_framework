const Optimizer = require("./optimizer")
const PropagationMath = require("../propagation_math");
const ActivationFunctions = require("../activation_functions");
const LossFunctions = require("../loss_functions");

class Adam extends Optimizer {
    constructor() {
        super();
        this.alpha = 0.001;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 10E-8;
    }
    SetAlpha(alpha) {
        this.alpha = alpha;
        return this;
    }
    SetBeta1(beta1) {
        this.beta1 = beta1;
        return this;
    }
    SetBeta2(beta2) {
        this.beta2 = beta2;
        return this;
    }
    SetEpsilon(epsilon) {
        this.epsilon = epsilon;
        return this;
    }

    Backpropagation(inputs, y_true, layer_sizes, weights, bias, activations, loss_function, recurrent_weights=undefined) {
        let m_weights = weights.map((layer) => {
            return Array(layer.length).fill(0);
        });
        let v_weights = weights.map((layer) => {
            return Array(layer.length).fill(0);
        });

        let m_bias = bias.map((layer) => {
            return Array(layer.length).fill(0);
        });
        let v_bias = bias.map((layer) => {
            return Array(layer.length).fill(0);
        });

        let m_reccurant = recurrent_weights === undefined ? undefined : recurrent_weights.map((layer) => {
            return Array(layer.length).fill(0);
        });
        let v_reccurant = recurrent_weights === undefined ? undefined : recurrent_weights.map((layer) => {
            return Array(layer.length).fill(0);
        });

        let new_weights = weights;
        let new_bias = bias;
        let new_reccurant_weights = recurrent_weights;

        let t = 0;

        let loss = 0;
        let loss_counter = 0;

        let last_logged_batch = -1;
        for (let epoch = 0; epoch < this.epochs; epoch++) {
            loss /= loss_counter === 0 ? 1 : loss_counter;
            console.log("EPOCH", epoch, " / ", this.epochs, "LOSS:", loss);
            t++;
            let batch_counter = 0;
            let batch = 0;
            loss_counter = 0;
            loss = 0;
            let gradiant = {};
            gradiant.weight_gradiants = weights.map((weight_g_l, layer_index) => {
                return weight_g_l.map((weight, weight_index) => {
                    return 0
                })
            });
            gradiant.bias_gradiants = bias.map((weight_g_l, layer_index) => {
                return weight_g_l.map((weight, weight_index) => {
                    return 0
                })
            });
            gradiant.reccurant_weight_gradiants = recurrent_weights.map((weight_g_l, layer_index) => {
                return weight_g_l.map((weight, weight_index) => {
                    return 0
                })
            });
            
            for (let training_counter = 0; training_counter < inputs.length; training_counter++) {
                let forward_prop_neurons = PropagationMath.ForwardPropagation(
                    inputs[training_counter],
                    weights,
                    bias,
                    activations,
                    layer_sizes,
                    recurrent_weights
                );
                loss_counter++;
                loss += forward_prop_neurons.map((d) => {
                    return d[0].map((n) => {
                        return LossFunctions.Loss(loss_function, y_true[training_counter], n);
                    }).reduce((partialSum, a) => partialSum + a, 0);
                }).reduce((partialSum, a) => partialSum + a, 0);

                let next_gradiant = PropagationMath.CalculateGradiant(
                    forward_prop_neurons,
                    weights,
                    bias,
                    y_true[training_counter],
                    loss_function,
                    activations,
                    recurrent_weights
                )

                gradiant.weight_gradiants = gradiant.weight_gradiants.map((weight_g_l, layer_index) => {
                    return weight_g_l.map((weight, weight_index) => {
                        return weight + next_gradiant.weight_gradiants[layer_index][weight_index];
                    })
                });
                gradiant.bias_gradiants = gradiant.bias_gradiants.map((weight_g_l, layer_index) => {
                    return weight_g_l.map((weight, weight_index) => {
                        return weight + next_gradiant.bias_gradiants[layer_index][weight_index];
                    })
                });
                gradiant.reccurant_weight_gradiants = gradiant.reccurant_weight_gradiants.map((weight_g_l, layer_index) => {
                    return weight_g_l.map((weight, weight_index) => {
                        return weight + next_gradiant.reccurant_weight_gradiants[layer_index][weight_index];
                    })
                });

                if (batch_counter >= this.batch_size) {
                    batch_counter = 0;
                    batch++;
                    let weight_gradiants = gradiant.weight_gradiants;
                    let bias_gradiants = gradiant.bias_gradiants;
                    let reccurant_weight_gradiants = gradiant.reccurant_weight_gradiants;

                    for (let layer = 0; layer < weights.length; layer++) {
                        if (layer !== weights.length - 1) {
                            for (let weight = 0; weight < weights[layer].length; weight++) {
                                m_weights[layer][weight] = this.beta1 * m_weights[layer][weight] + (1 - this.beta1) * weight_gradiants[layer][weight];
                                v_weights[layer][weight] = this.beta2 * v_weights[layer][weight] + (1 - this.beta2) * Math.pow(weight_gradiants[layer][weight], 2);
                                let m_hat = m_weights[layer][weight] / (1 - Math.pow(this.beta1, t));
                                let v_hat = v_weights[layer][weight] / (1 - Math.pow(this.beta2, t));
                                new_weights[layer][weight] = new_weights[layer][weight] + this.alpha * m_hat / (Math.sqrt(v_hat) + this.epsilon);
                            }
                        }

                        if (layer !== 0) {
                            for (let _bias = 0; _bias < bias[layer].length; _bias++) {
                                m_bias[layer][_bias] = this.beta1 * m_bias[layer][_bias] + (1 - this.beta1) * bias_gradiants[layer][_bias];
                                v_bias[layer][_bias] = this.beta2 * v_bias[layer][_bias] + (1 - this.beta2) * Math.pow(bias_gradiants[layer][_bias], 2);
                                let m_hat = m_bias[layer][_bias] / (1 - Math.pow(this.beta1, t));
                                let v_hat = v_bias[layer][_bias] / (1 - Math.pow(this.beta2, t));
                                new_bias[layer][_bias] = new_bias[layer][_bias] + this.alpha * m_hat / (Math.sqrt(v_hat) + this.epsilon);
                            }
                        }

                        if (layer !== weights.length - 1 || layer !== 0) {
                            for (let reccurant_weight = 0; reccurant_weight < recurrent_weights[layer].length; reccurant_weight++) {
                                m_reccurant[layer][reccurant_weight] = this.beta1 * m_reccurant[layer][reccurant_weight] + (1 - this.beta1) * reccurant_weight_gradiants[layer][reccurant_weight];
                                v_reccurant[layer][reccurant_weight] = this.beta2 * v_reccurant[layer][reccurant_weight] + (1 - this.beta2) * Math.pow(reccurant_weight_gradiants[layer][reccurant_weight], 2);
                                let m_hat = m_reccurant[layer][reccurant_weight] / (1 - Math.pow(this.beta1, t));
                                let v_hat = v_reccurant[layer][reccurant_weight] / (1 - Math.pow(this.beta2, t));
                                new_reccurant_weights[layer][reccurant_weight] = new_reccurant_weights[layer][reccurant_weight] + this.alpha * m_hat / (Math.sqrt(v_hat) + this.epsilon);
                            }
                        }
                    }
                }
                batch_counter++;
            }
        }
        return {
            weight_gradiants: new_weights,
            bias_gradiants: new_bias,
            reccurant_weight_gradiants: new_reccurant_weights
        }
    }
}

module.exports = Adam;