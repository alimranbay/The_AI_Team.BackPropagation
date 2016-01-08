package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.BackPropagationNetwork;

/**
 * A controller class that controls input and output
 */
public class ViewController {
    private BackPropagationNetwork backPropagationNetwork;

    public ViewController(BackPropagationNetwork backPropagationNetwork) {
        this.backPropagationNetwork = backPropagationNetwork;
    }

    public void initializeNetwork(Double learningRate, Double threshhold) {
        backPropagationNetwork.initialize(learningRate,threshhold);
    }

    public double[] getInputValues(){
        return backPropagationNetwork.getInputCells();
    }

    public double[] getTargetValues() {
        return backPropagationNetwork.getTargets();
    }

    public double[] getHiddenValues() {
        return backPropagationNetwork.getHiddenCells();
    }

    public double[] getOutputCells() {
        return backPropagationNetwork.getOutputCells();
    }

    public double[][] getIhWeights() {
        return backPropagationNetwork.getIhWeights();
    }

    public double[][] getHoWeights() {
        return backPropagationNetwork.getHoWeights();
    }

    public int getEpoch() {
        return backPropagationNetwork.getEpoch();
    }
}
