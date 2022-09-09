class MatrixMath {
    static GetWeightIndexBetweenNodes(layer2_length, node1, node2) {
        return layer2_length*node1 + node2;
    }
}

module.exports = MatrixMath;