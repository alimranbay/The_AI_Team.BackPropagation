package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.NeuralNetwork;

/**
 * This class
 */
public class Controller {
    private NeuralNetwork neuralNetwork;

    public Controller(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public void startNeuralNetwork() {
        computeOutputs();
        computeErrors();
    }

    public double[] computeOutputs(){
        int numberOfInputCells = neuralNetwork.getNumberOfInputCells();
        int numberOfHiddenCells = neuralNetwork.getNumberOfHiddenCells();
        int numberOfOutputCells = neuralNetwork.getNumberOfOutputCells();
        // initialize hidden cells
        for (int i = 0; i < numberOfHiddenCells; i++)
            neuralNetwork.setHiddenCell(i, 0);

        for (int i = 0; i < numberOfHiddenCells; i++) {
            for (int j = 0; j < numberOfInputCells; j++){
                neuralNetwork.setHiddenCell(i,
                        neuralNetwork.getHiddenCells()[i] +
                        (neuralNetwork.getInputCells()[j] *
                        neuralNetwork.getIhWeights()[j][i]));
            }
            neuralNetwork.setHiddenCell(i, neuralNetwork.getHiddenCells()[i] + neuralNetwork.gethBiases()[i]);
        }


        double[] tempHiddens = new double[numberOfHiddenCells];
        for (int i = 0; i < numberOfHiddenCells; i++)
            tempHiddens[i] = hyperTanFunction(neuralNetwork.getHiddenCells()[i]);

        double outputCells[] = neuralNetwork.getOutputCells();
        for (int i = 0; i < numberOfOutputCells; i++) {
            for (int j = 0; j < numberOfHiddenCells; j++)
                outputCells[i] += (tempHiddens[j] * neuralNetwork.getHoWeights()[j][i]);
            outputCells[i] += neuralNetwork.getoBiases()[i];
        }

        for (int i = 0; i < numberOfOutputCells; i++)
            outputCells[i] = sigmoidFunction(outputCells[i]);
        neuralNetwork.setOutputCells(outputCells);

        return outputCells;
    }

    public double[] computeErrors() {
        double[] errors = neuralNetwork.getErrors();
        for (int i = 0; i < neuralNetwork.getNumberOfOutputCells(); i++)
            errors[i] = Math.abs(neuralNetwork.getTargets()[i] - neuralNetwork.getOutputCells()[i]);

        neuralNetwork.setErrors(errors);
        return errors;
    }

    private static double hyperTanFunction(double x)
    {
        if (x < -45.0)
            return -1.0;
        if (x > 45.0)
            return 1.0;
        return Math.tanh(x);
    }

    private static double sigmoidFunction(double x)
    {
        if (x < -45.0)
            return 0.0;
        if (x > 45.0)
            return 1.0;
        return 1.0/(1.0 + Math.exp(-x));
    }

}
