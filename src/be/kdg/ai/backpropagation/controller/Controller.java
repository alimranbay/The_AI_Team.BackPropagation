package be.kdg.ai.backpropagation.controller;

/**
 * This class is the controller for neural networks
 */
public interface Controller {
    void stopBackpropagation();

    void startBackpropagation();

    /**
     * Computes the output of the backpropagation network
     * @return the output
     */
    double[] computeOutputs();

    /**
     * Computes the error between the target and the output of the backpropagation network
     * @return the error
     */
    double[] computeErrors();

    /**
     * Updates the weights between input, output and hidden cells
     */
    void updateWeights();
}
