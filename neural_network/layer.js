const ActivationFunctions = require("./math_functions/activation_functions");

class Layer {
    constructor() {
        this.size = 0;
        this.recurrent = false;
        this.dropout = 0;
        this.activation = ActivationFunctions.Activations.relu;
    }

    SetSize(size) {
        this.size = size;
        return this;
    }

    SetDropout(percentage) {
        this.dropout = percentage;
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

    SetActivation(activation) {
        this.activation = activation;
        return this;
    }
}

module.exports = Layer;