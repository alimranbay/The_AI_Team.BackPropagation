package be.kdg.ai.backpropagation.controller.algorithm;

/**
 * An interface that's going to be implemented by a neural network algorithm.
 */
public interface NeuralNetwork {
    double[] computeOutputs();

    double[] computeErrors();

    double[] getInputCells();

    double[] getTargets();
}
