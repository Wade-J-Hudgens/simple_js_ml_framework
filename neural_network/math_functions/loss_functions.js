const { getToPathname } = require("react-router/lib/router");

class LossFunctions {
    static Losses = {
        BinaryCrossentropy: 0,
        BinaryFocalCrossentropy: 1,
        CategoricalCrossentropy: 2,
        CategoricalHinge: 3,
        Hinge: 5,
        Huber: 6,
        KLDivergence: 7,
        LogCosh: 8,
        MeanAbsoluteError: 9,
        MeanAbsolutePercentageError: 10,
        MeanSquaredError: 11,
        Poisson: 12,
        SquaredHinge: 15
    }

    static BinaryCrossentropy(y_true, y_pred, prime=false) {
        if (prime) {
            return (y_true/y_pred) * (-1/(1-y_pred))
        }
        return (y_true * Math.log(y_pred) + (1-y_true)) * Math.log(1-y_pred);
    }

    static BinaryFocalCrossentropy(y_true, y_pred, prime=false) {
        if (prime) {
            return y_true === 1 ?
            -1/y_pred :
            1/(1-y_pred)
        }
        return y_true === 1 ?
        -Math.log(y_pred) :
        -Math.log(1 - y_pred)
    }

    static CategoricalCrossentropy(y_true, y_pred, prime=false) {
        if (prime) {
            return y_true/y_pred;
        }
        return y_true*Math.log(y_pred)
    }

    static CategoricalHinge(y_true, y_pred, prime=false) {
        if (prime) {
            return Math.max(0, 1-y_true);
        }
        return Math.max(0, 1 - y_pred*y_true);
    }

    static Hinge(y_true, y_pred, prime=false) {
        if (prime) {
            return 1-y_pred*y_true <= 0 ? 0 : y_true;
        }

        return Math.max(0, 1-y_pred*y_true);
    }

    static Huber(y_true, y_pred, prime=false, huber_delta=0.5) {
        if (prime) {
            Math.pow(y_true - y_pred, 2) <= huber_delta ? y_true - y_pred : (huber_delta*(-y_pred + y_true))/(-y_pred + y_true)
        }
        return Math.pow(y_true - y_pred, 2) <= huber_delta ? 0.5 * Math.pow(y_true - y_pred, 2) : huber_delta * (Math.abs(y_true - y_pred) - 0.5 * huber_delta)
    }

    static KLDivergence(y_true, y_pred, prime=false) {
        if (prime) {
            return Math.log(y_pred/y_true) + 1
        }
        return y_pred * Math.log(y_pred/y_true);
    }

    static LogCosh(y_true, y_pred, prime=false) {
        if (prime) {
            Math.tanh(x);
        }
        return Math.log(Math.cosh(y_pred - y_true));
    }

    static MeanAbsoluteError(y_true, y_pred, prime=false) {
        if (prime) {
            return (-y_pred + y_true)/(Math.abs(-y_pred + y_true));
        }
        return Math.abs(y_true - y_pred);
    }

    static MeanAbsolutePercentageError(y_true, y_pred, prime=false) {
        if (prime) {
            return y_true/Math.pow(y_pred, 2);
        }
        return (y_pred - y_true) / y_pred;
    }

    static MeanSquaredError(y_true, y_pred, prime=false) {
        if (prime) {
            return 2 * (y_true - y_pred)
        }
        return Math.pow(y_true - y_pred, 2);
    }

    static Poisson(y_true, y_pred, prime=false) {
        if (prime) {
            return 1 - y_true/y_pred;
        }
        return y_pred - y_true * Math.log(y_pred);
    }

    static SquaredHinge(y_true, y_pred, prime=false) {
        if (prime) {
            return 1 - y_pred*y_true <= 0 ? 0 : 2 * y_true
        }
        return Math.pow(Math.max(0, 1 - y_pred*y_true), 2);
    }

    static Loss(loss_function, y_true, y_pred, prime=false, huber_delta=0.5) {
        switch (loss_function) {
            case LossFunctions.Losses.BinaryCrossentropy:
                return this.BinaryCrossentropy(y_true, y_pred, prime);
            case LossFunctions.Losses.BinaryFocalCrossentropy:
                return this.BinaryFocalCrossentropy(y_true, y_pred, prime);
            case LossFunctions.Losses.CategoricalCrossentropy:
                return this.CategoricalCrossentropy(y_true, y_pred, prime);
            case LossFunctions.Losses.CategoricalHinge:
                return this.CategoricalHinge(y_true, y_pred, prime);
            case LossFunctions.Losses.Hinge:
                return this.Hinge(y_true, y_pred, prime);
            case LossFunctions.Losses.Huber:
                return this.Huber(y_true, y_pred, prime, huber_delta);
            case LossFunctions.Losses.KLDivergence:
                return this.KLDivergence(y_true, y_pred, prime);
            case LossFunctions.Losses.LogCosh:
                return this.LogCosh(y_true, y_pred, prime);
            case LossFunctions.Losses.MeanAbsoluteError:
                return this.MeanAbsoluteError(y_true, y_pred, prime);
            case LossFunctions.Losses.MeanAbsolutePercentageError:
                return this.MeanAbsolutePercentageError(y_true, y_pred, prime);
            case LossFunctions.Losses.MeanSquaredError:
                return this.MeanSquaredError(y_true, y_pred, prime);
            case LossFunctions.Losses.Poisson:
                return this.Poisson(y_true, y_pred, prime);
            case LossFunctions.Losses.SquaredHinge:
                return this.SquaredHinge(y_true, y_pred, prime);
        }
    }
}

module.exports = LossFunctions;