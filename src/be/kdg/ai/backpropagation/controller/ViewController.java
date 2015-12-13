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

    public void initializeNetwork() {
        neuralNetwork.initialize();
    }

    public double[] getInputValues(){
        return neuralNetwork.getInputCells();
    }

    public double[] getTargetValues() {
        return neuralNetwork.getTargets();
    }

    public double[] getHiddenValues() {
        return neuralNetwork.getHiddenCells();
    }

    public double[] getOutputCells() {
        return neuralNetwork.computeOutputs();
    }

    public double[][] getIhWeights() {
        return neuralNetwork.getIhWeights();
    }

    public double[][] getHoWeights() {
        return neuralNetwork.getHoWeights();
    }

    public double[] getErrors() {
        return neuralNetwork.computeErrors();
    }
}
