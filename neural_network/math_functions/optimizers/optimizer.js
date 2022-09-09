class Optimizer {
    constructor() {
        this.batch_size = 1;
        this.epochs = 1;
    }

    Backpropagation(inputs, y_true, layer_sizes, weights, bias, recurrent_weights=undefined) {}
    SetBatchSize(batch_size) {
        this.batch_size = batch_size;
        return this;
    }
    SetEpochs(epochs) {
        this.epochs = epochs;
        return this;
    }
}

module.exports = Optimizer;