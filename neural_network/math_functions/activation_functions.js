class ActivationFunctions {
    static Activations = {
        elu: 0,
        sigmoid: 3,
        linear: 4,
        relu: 5,
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

    static Softplus(z, prime=false) {
        if (prime) {
            return (Math.exp(z)/(1+Math.exp(z)));
        }
        return Math.log(1 + Math.exp(z));
    }

    static Softsign(z, prime=false) {
        if (prime) {
            return 1/Math.pow(1 + Math.abs(z), 2);
        }
        return z / (1 + Math.abs(z));
    }

    static Swish(z, prime=false) {
        if (prime) {
            return ActivationFunctions.Swish(z) * ActivationFunctions.Sigmoid(z) * (1- ActivationFunctions.Swish(z));
        }

        return z * ActivationFunctions.Sigmoid(z);
    }

    static tanh(z, prime=false) {
        if (prime) {
            return 4 / Math.pow(Math.exp(z) + Math.exp(-z), 2);
        }
        return (Math.exp(z) - Math.exp(-z))/(Math.exp(z) + Math.exp(-z))
    }

    static Activation(activation_function, z, prime=false) {
        switch(activation_function) {
            case ActivationFunctions.Activations.relu:
                return ActivationFunctions.Relu(z, prime);
            case ActivationFunctions.Activations.lrelu:
                return ActivationFunctions.LRelu(z, prime);
            case ActivationFunctions.Activations.linear:
                return ActivationFunctions.Linear(z, prime);
            case ActivationFunctions.Activation.elu:
                return ActivationFunctions.Elu(z, prime);
            case ActivationFunctions.Activation.elu:
                return ActivationFunctions.elu(z, prime);
            case ActivationFunctions.Activation.softplus:
                return ActivationFunctions.Softplus(z, prime);
            case ActivationFunctions.Activation.softsign:
                return ActivationFunctions.Softsign(z, prime);
            case ActivationFunctions.Activation.swish:
                return ActivationFunctions.Swish(z, prime);
            case ActivationFunctions.Activation.tanh:
                return ActivationFunctions.tanh(z, prime);
        }
    }
}

module.exports = ActivationFunctions;