package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.NeuralNetwork;

/**
 * A controller class that controls input and output
 */
public class ViewController {
    private NeuralNetwork neuralNetwork;
    private Controller controller;

    public ViewController(NeuralNetwork neuralNetwork, Controller controller) {
        this.neuralNetwork = neuralNetwork;
        this.controller = controller;
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
        return controller.computeOutputs();
    }

    public double[][] getIhWeights() {
        return neuralNetwork.getIhWeights();
    }

    public double[][] getHoWeights() {
        return neuralNetwork.getHoWeights();
    }

    public double[] getErrors() {
        return controller.computeErrors();
    }
}
