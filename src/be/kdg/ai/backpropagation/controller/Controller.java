package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.controller.algorithm.NeuralNetwork;

/**
 * This class
 */
public class Controller {
    private NeuralNetwork neuralNetwork;

    public Controller(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public void startNeuralNetwork() {
        neuralNetwork.computeOutputs();
        neuralNetwork.computeErrors();
    }
}
