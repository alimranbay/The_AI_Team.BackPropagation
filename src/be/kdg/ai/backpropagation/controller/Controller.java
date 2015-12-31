package be.kdg.ai.backpropagation.controller;

/**
 * This class is the controller for neural networks
 */
public interface Controller {
    void startBackpropagation();

    double[] computeOutputs();

    double[] computeErrors();

    void updateWeights();
}
