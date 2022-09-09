class ActivationFunctions {
    static Activations = {
        elu: 0,
        exponential: 1,
        gelu: 2,
        sigmoid: 3,
        linear: 4,
        relu: 5,
        selu: 6,
        softmax: 7,
        softplus: 8,
        softsign: 9,
        swish: 10,
        tanh: 11,
        lrelu: 12
    }

    static Elu(z, prime=false) {
        if (prime) {
            return Math.exp(z);
        }
        return Math.exp(z) - 1;
    }

    static Exponential(z, prime=false) {
         
    }

    static Gelu(z, prime=false) {

    }

    static Sigmoid(z, prime=false) {
        if (prime) {
            return Math.exp(-z) / (Math.pow(1 + Math.exp(-z), 2));
        }
        return 1 / (1 + Math.exp(-z));
    }

    static Linear(z, prime=false) {
        if (prime) {
            return 1;
        }
        return z;
    }

    static Relu(z, prime=false) {
        if (prime) {
            return z <= 0 ? 0 : 1;
        }
        return z <= 0 ? 0 : z;
    }

    static LRelu(z, prime=false) {
        if (prime) {
            return z<=0?0.01*z:1;
        }
        return z<=0?0.01:z;
    }

    static Activation(activation_function, z, prime=false) {
        switch(activation_function) {
            case ActivationFunctions.Activations.relu:
                return ActivationFunctions.Relu(z, prime);
            case ActivationFunctions.Activations.lrelu:
                return ActivationFunctions.LRelu(z, prime);
            case ActivationFunctions.Activations.linear:
                return ActivationFunctions.Linear(z, prime);
        }
    }
}

module.exports = ActivationFunctions;