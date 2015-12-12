package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.controller.algorithm.NeuralNetwork;

/**
 * A controller class that controls input and output
 */
public class ViewController {
    private NeuralNetwork neuralNetwork;

    public ViewController(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public double[] getInputValues(){
        return neuralNetwork.getInputCells();
    }

    public double[] getTargetValues() {
        return neuralNetwork.getTargets();
    }
}
