package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.BackPropagationNetwork;

/**
 * A controller class that controls input and output
 */
public class ViewController {
    private BackPropagationNetwork backPropagationNetwork;
    private Controller controller;

    public ViewController(BackPropagationNetwork backPropagationNetwork, Controller controller) {
        this.backPropagationNetwork = backPropagationNetwork;
        this.controller = controller;
    }

    public void initializeNetwork() {
        backPropagationNetwork.initialize();
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
        return controller.computeOutputs();
    }

    public double[][] getIhWeights() {
        return backPropagationNetwork.getIhWeights();
    }

    public double[][] getHoWeights() {
        return backPropagationNetwork.getHoWeights();
    }
}
