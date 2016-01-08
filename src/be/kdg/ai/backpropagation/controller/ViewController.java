package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.BackPropagationNetwork;
import be.kdg.ai.backpropagation.model.InitialisationException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * A controller class that controls input and output
 */
public class ViewController {
    private BackPropagationNetwork backPropagationNetwork;
    private final static Logger logger = LogManager.getLogger(ViewController.class);

    public ViewController(BackPropagationNetwork backPropagationNetwork) {
        this.backPropagationNetwork = backPropagationNetwork;
    }

    public void initializeNetwork(Double learningRate, Double threshhold) {
        try {
            backPropagationNetwork.initialize(learningRate,threshhold);
        } catch (InitialisationException e) {
            logger.error(e);
        }
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
