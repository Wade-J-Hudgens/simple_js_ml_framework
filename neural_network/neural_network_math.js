const ActivationFunctions = require("./math_functions/activation_functions");
const LossFunctions = require("./math_functions/loss_functions");
const Optimizers = require("./math_functions/optimizers")
const PropagationMath = require("./math_functions/propagation_math")

class NeuralNetworkMath {
    static Activations = ActivationFunctions.Activations;
    static Losses = LossFunctions.Losses;
    static Optimizers = Optimizers.Optimizers;
    static PropagationMath = PropagationMath;
}

module.exports = NeuralNetworkMath;