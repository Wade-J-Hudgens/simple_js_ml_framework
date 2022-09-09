const ActivationFunctions = require("./activation_functions");
const LossFunctions = require("./loss_functions");
const MatrixMath = require("./matrix_math");

class PropagationMath {
    static dCx_da() {

    }

    static dCx_dw() {

    }

    static dCx_db() {

    }

    static dz_dw() {

    }

    static da_dz() {

    }

    static dz_da() {

    }

    static dz_db() {

    }

    /**
     * 
     * @param {number[][]} neurons 
     * @param {number[][]} weights 
     * @param {number[][]} bias 
     * @param {number[]} y_true 
     * @param {number[][]} recurrent_weights
     * @param {LossFunctions.Losses} loss_function 
     */
    static CalculateGradiant(neurons, weights, bias, y_true, loss_function, activations, recurrent_weights=undefined) {
        let memo_dcx_da = {};

        let memo_da_dz = {};

        let weight_gradiants = weights.map((layer)=>{
            return Array(layer.length).fill(0);
        });

        let bias_gradiants = bias.map((layer) => {
            return Array(layer.length).fill(0);
        });

        let reccurant_weight_gradiants = recurrent_weights === undefined ? undefined :
        recurrent_weights.map((layer) => {
            return Array(layer.length).fill(0);
        })

        let f = true;
        let c = 0;
        time_loop:
        for (let time = neurons.length - 1; time >= 0; time--) {
            layer_loop:
            for (let layer = neurons[time].length - 1; layer >= 0; layer--) {
                //Calculating dcxda
                neuron_loop:
                for (let neuron = 0; neuron < neurons[time][layer].length; neuron++) {
                    if (layer === neurons[time].length - 1) {
                        memo_dcx_da[`${time}_${layer}_${neuron}`] = LossFunctions.Loss(loss_function, y_true[time][neuron], neurons[time][layer][neuron], true);
                        continue neuron_loop;
                    }
                    let dCx_da = 0;
                    let next_layer = layer + 1;
                    for (let next_neuron = 0; next_neuron < neurons[time][next_layer].length; next_neuron++) {
                        let weight_index = MatrixMath.GetWeightIndexBetweenNodes(neurons[time][next_layer].length, neuron, next_neuron);
                        let dz_da = weights[layer][weight_index];
                        //let da_dz = memo_da_dz[`${time}_${next_layer}_${next_neuron}`] === undefined ? ActivationFunctions.Activation(activations[next_layer], neurons[time][next_layer][next_neuron], true) : memo_da_dz[`${time}_${next_layer}_${next_neuron}`];
                        let da_dz = ActivationFunctions.Activation(activations[next_layer], neurons[time][next_layer][next_neuron], true);
                        memo_da_dz[`${time}_${next_layer}_${next_neuron}`] = da_dz;
                        let dcx_da_next = memo_dcx_da[`${time}_${next_layer}_${next_neuron}`];
                        dCx_da += dz_da * da_dz * dcx_da_next;
                    }
                    memo_dcx_da[`${time}_${layer}_${neuron}`] = dCx_da;
                }
                //Calculating gradiant
                if (layer !== neurons[time].length - 1) {
                    first_neuron_loop:
                    for (let first_neuron = 0; first_neuron < neurons[time][layer].length; first_neuron++) {
                        second_neuron_loop:
                        for (let second_neuron = 0; second_neuron < neurons[time][layer+1].length; second_neuron++) {
                            let weight_index = MatrixMath.GetWeightIndexBetweenNodes(neurons[time][layer+1].length, first_neuron, second_neuron);
                            let dz_dw = neurons[time][layer][first_neuron];
                            //let da_dz = memo_da_dz[`${time}_${layer+1}_${second_neuron}`];
                            let da_dz = ActivationFunctions.Activation(activations[layer+1], neurons[time][layer+1][second_neuron], true);
                            let dcx_da = memo_dcx_da[`${time}_${layer+1}_${second_neuron}`];
                            
                            let weight_gradiant = dz_dw * da_dz * dcx_da;
                            weight_gradiants[layer][weight_index] = weight_gradiant;
                        }
                    }

                    for (let next_neuron = 0; next_neuron < neurons[time][layer+1].length; next_neuron++) {
                        let dz_db = bias[layer+1][next_neuron];
                        //let da_dz = memo_da_dz[`${time}_${layer+1}_${next_neuron}`];
                        let da_dz = ActivationFunctions.Activation(activations[layer+1], neurons[time][layer+1][next_neuron], true);
                        let dcx_da = memo_dcx_da[`${time}_${layer+1}_${next_neuron}`];
                        let bias_gradiant = dz_db * da_dz * dcx_da;
                        bias_gradiants[layer+1][next_neuron] = bias_gradiant;

                        if (neurons.length > 1 && time !== 0 && layer !== 0 && layer !== neurons.length - 1) {
                            for (let previous_neuron = 0; previous_neuron < neurons[time-1][layer+1].length; previous_neuron++) {
                                let weight_index = MatrixMath.GetWeightIndexBetweenNodes(neurons[time][layer+1].length, previous_neuron, next_neuron);
                                let dz_dw = neurons[time-1][layer+1][previous_neuron];
                                //let da_dz = memo_da_dz[`${time}_${layer+1}_${next_neuron}`];
                                let da_dz = ActivationFunctions.Activation(activations[layer+1], neurons[time][layer+1][next_neuron], true);
                                let dcx_da = memo_dcx_da[`${time}_${layer+1}_${next_neuron}`];
                                let reccurant_weight_gradiant = dz_dw * da_dz * dcx_da;
                                reccurant_weight_gradiants[layer+1][weight_index] = reccurant_weight_gradiant;
                            } 
                        }
                    }
                }
            }
        }
        return {
            weight_gradiants: weight_gradiants,
            bias_gradiants: bias_gradiants,
            reccurant_weight_gradiants: reccurant_weight_gradiants
        }
    }

    /**
     * 
     * @param {number[][]} inputs 
     * @param {number[][]} weights 
     * @param {bias[][]} bias 
     * @param {ActivationFunctions.Activations} activations 
     * @param {number[][]} recurrent_weights 
     */
    static ForwardPropagation(inputs, weights, bias, activations, layer_sizes, recurrent_weights=undefined, p=false) {
        let neurons = Array(inputs.length).fill().map(()=>{
            return layer_sizes.map((layer_size) => {
                return Array(layer_size).fill(0);
            })
        });
        for (let time = 0; time < inputs.length; time++) {
            neurons[time][0] = inputs[time];
            for (let layer = 1; layer < neurons[time].length; layer++) {
                for (let neuron = 0; neuron < neurons[time][layer].length; neuron++) {
                    let z = 0;
                    for (let previous_neuron = 0; previous_neuron < neurons[time][layer-1].length; previous_neuron++) {
                        let weight_index = MatrixMath.GetWeightIndexBetweenNodes(neurons[time][layer].length, previous_neuron, neuron);
                        let weight_value = weights[layer-1][weight_index];
                        let activation_value = neurons[time][layer-1][previous_neuron];
                        z += activation_value * weight_value;
                        
                    }
                    
                    if (time !== 0 && layer !== 0 && layer !== neurons.length - 1 && recurrent_weights[layer].length > 0) {
                        for (let previous_neuron = 0; previous_neuron < neurons[time-1][layer].length; previous_neuron++) {
                            let weight_index = MatrixMath.GetWeightIndexBetweenNodes(neurons[time][layer].length, previous_neuron, neuron);
                            let weight_value = recurrent_weights[layer][weight_index];
                            let activation_value = neurons[time-1][layer][neuron];
                            z += activation_value * weight_value;
                        }
                    }

                    let bias_value = bias[layer][neuron];
                    z += bias_value;
                    let activation = ActivationFunctions.Activation(
                        activations[layer],
                        z
                    )
                    neurons[time][layer][neuron] = activation;
                }
            }
        }
        return neurons;
    }
}

module.exports = PropagationMath;