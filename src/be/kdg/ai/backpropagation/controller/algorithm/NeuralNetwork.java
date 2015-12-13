package be.kdg.ai.backpropagation.controller.algorithm;

/**
 * An interface that's going to be implemented by a neural network algorithm.
 */
public interface NeuralNetwork {
    void initialize();

    double[] computeOutputs();

    double[] computeErrors();

    double[] getInputCells();

    double[] getTargets();

    double[] getHiddenCells();

    public double[][] getIhWeights();


    public double[][] getHoWeights();

}
