package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.NeuralNetwork;

/**
 * This class
 */
public class Controller {
    private NeuralNetwork neuralNetwork;
    private double[] tempHiddens;

    public Controller(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
        tempHiddens = new double[neuralNetwork.getNumberOfHiddenCells()];
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

    public void updateWeights () {
        double[] outputCells = neuralNetwork.getOutputCells();
        double[] inputCells = neuralNetwork.getInputCells();
        double[] hiddenCells = neuralNetwork.getHiddenCells();

        double[] targetValues = neuralNetwork.getTargets();
        double[][] hoWeights = neuralNetwork.getHoWeights();
        double[][] ihWeights = neuralNetwork.getIhWeights();

        double[] hiddenBiases = neuralNetwork.gethBiases();
        double[] outputBiases = neuralNetwork.getoBiases();

        double[] outputGradients = neuralNetwork.getOutputGradients();
        double[] hiddenGradients = neuralNetwork.getHiddenGradients();

        double derivative;
        double sum = 0.0;
        double delta;

        int numberOfOutputcells = neuralNetwork.getNumberOfOutputCells();
        int numberOfHiddenCells = neuralNetwork.getNumberOfHiddenCells();
        int numberOfInputCells = neuralNetwork.getNumberOfInputCells();

        double learningRate = neuralNetwork.getLEARNING_RATE();
        double momentum = neuralNetwork.getMomentum();

        double[][] ihPreviousWeightsDelta = neuralNetwork.getIhPreviousWeightsDelta();
        double[] hPreviousBiasesDelta = neuralNetwork.gethPreviousBiasesDelta();
        double[][] hoPreviousWeightsDelta = neuralNetwork.getHoPreviousWeightsDelta();
        double[] oPreviousBiasesDelta = neuralNetwork.getoPreviousBiasesDelta();

        // Calculate gradients
        for (int i = 0; i < numberOfOutputcells; i++) {
            derivative = (1 - outputCells[i]) * outputCells[i];
            outputGradients[i] = derivative * (targetValues[i] - outputCells[i]);
        }
        neuralNetwork.setOutputGradients(outputGradients);

        for (int i = 0; i < numberOfHiddenCells; i++) {
            derivative = (1 - tempHiddens[i]) * (1 + tempHiddens[i]);
            for (int j = 0; j < neuralNetwork.getNumberOfOutputCells(); j++) {
                sum += outputGradients[j] * hoWeights[i][j];
            }
            hiddenGradients[i] = derivative * sum;
        }
        neuralNetwork.setHiddenGradients(hiddenGradients);

        // update weights & biases
        for (int i = 0; i < numberOfInputCells; i++)
            for (int j = 0; j < numberOfHiddenCells; j++) {
                delta = learningRate * hiddenGradients[j] * inputCells[i];
                ihWeights[i][j] += delta;
                ihWeights[i][j] += momentum * ihPreviousWeightsDelta[i][j];
                ihPreviousWeightsDelta[i][j] = delta;
            }
        neuralNetwork.setIhWeights(ihWeights);
        neuralNetwork.setIhPreviousWeightsDelta(ihPreviousWeightsDelta);

        for (int i = 0; i < numberOfHiddenCells; i++){
            delta = learningRate * hiddenGradients[i];
            hiddenBiases[i] += delta;
            hiddenBiases[i] += momentum * hPreviousBiasesDelta[i];
            hPreviousBiasesDelta[i] = delta;
        }
        neuralNetwork.sethBiases(hiddenBiases);
        neuralNetwork.sethPreviousBiasesDelta(hPreviousBiasesDelta);

        for (int i = 0; i < numberOfHiddenCells; i++)
            for (int j = 0; j < numberOfOutputcells; j++) {
                delta = learningRate * outputGradients[j] * hiddenCells[i];
                hoWeights[i][j] += delta;
                hoWeights[i][j] += momentum * hoPreviousWeightsDelta[i][j];
                hoPreviousWeightsDelta[i][j] = delta;
            }
        neuralNetwork.setHoWeights(hoWeights);
        neuralNetwork.setHoPreviousWeightsDelta(hoPreviousWeightsDelta);

        for (int i = 0; i < numberOfOutputcells; i++){
            delta = learningRate * outputGradients[i];
            outputBiases[i] += delta;
            outputBiases[i] += momentum * oPreviousBiasesDelta[i];
            oPreviousBiasesDelta[i] = delta;
        }
        neuralNetwork.setoBiases(outputBiases);
        neuralNetwork.setoPreviousBiasesDelta(oPreviousBiasesDelta);
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
