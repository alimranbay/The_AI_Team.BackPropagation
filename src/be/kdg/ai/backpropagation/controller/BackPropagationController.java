package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.BackPropagationNetwork;
import be.kdg.ai.backpropagation.view.JavaFxView;

/**
 * This class
 */
public class BackPropagationController implements Controller {
    private BackPropagationNetwork backPropagationNetwork;
    private double[] tempHiddens;
    public BackPropagationController(BackPropagationNetwork backPropagationNetwork) {
        this.backPropagationNetwork = backPropagationNetwork;
        tempHiddens = new double[backPropagationNetwork.getNumberOfHiddenCells()];
    }

    @Override
    public void startBackpropagation() {
        for (int i = 0; i < backPropagationNetwork.getMAX_EPOCH(); i++) {
            computeOutputs();
            updateWeights();
            JavaFxView.changeValues();
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public double[] computeOutputs(){
        int numberOfInputCells = backPropagationNetwork.getNumberOfInputCells();
        int numberOfHiddenCells = backPropagationNetwork.getNumberOfHiddenCells();
        int numberOfOutputCells = backPropagationNetwork.getNumberOfOutputCells();
        // initialize hidden cells
        for (int i = 0; i < numberOfHiddenCells; i++)
            backPropagationNetwork.setHiddenCell(i, 0);

        for (int i = 0; i < numberOfHiddenCells; i++) {
            for (int j = 0; j < numberOfInputCells; j++){
                backPropagationNetwork.setHiddenCell(i,
                        backPropagationNetwork.getHiddenCells()[i] +
                        (backPropagationNetwork.getInputCells()[j] *
                        backPropagationNetwork.getIhWeights()[j][i]));
            }
            backPropagationNetwork.setHiddenCell(i, backPropagationNetwork.getHiddenCells()[i] + backPropagationNetwork.gethBiases()[i]);
        }

        for (int i = 0; i < numberOfHiddenCells; i++)
            tempHiddens[i] = hyperTanFunction(backPropagationNetwork.getHiddenCells()[i]);

        double outputCells[] = backPropagationNetwork.getOutputCells();
        for (int i = 0; i < numberOfOutputCells; i++) {
            for (int j = 0; j < numberOfHiddenCells; j++)
                outputCells[i] += (tempHiddens[j] * backPropagationNetwork.getHoWeights()[j][i]);
            outputCells[i] += backPropagationNetwork.getoBiases()[i];
        }

        for (int i = 0; i < numberOfOutputCells; i++)
            outputCells[i] = sigmoidFunction(outputCells[i]);
        backPropagationNetwork.setOutputCells(outputCells);

//        updateWeights();

        return outputCells;
    }

    @Override
    public double[] computeErrors() {
        double[] errors = backPropagationNetwork.getErrors();
        for (int i = 0; i < backPropagationNetwork.getNumberOfOutputCells(); i++)
            errors[i] = Math.abs(backPropagationNetwork.getTargets()[i] - backPropagationNetwork.getOutputCells()[i]);

        backPropagationNetwork.setErrors(errors);
        return errors;
    }

    @Override
    public void updateWeights() {
        double[] errors = computeErrors();

        double[] outputCells = backPropagationNetwork.getOutputCells();
        double[] inputCells = backPropagationNetwork.getInputCells();
        double[] hiddenCells = backPropagationNetwork.getHiddenCells();

        double[] targetValues = backPropagationNetwork.getTargets();
        double[][] hoWeights = backPropagationNetwork.getHoWeights();
        double[][] ihWeights = backPropagationNetwork.getIhWeights();

        double[] hiddenBiases = backPropagationNetwork.gethBiases();
        double[] outputBiases = backPropagationNetwork.getoBiases();

        double[] outputGradients = backPropagationNetwork.getOutputGradients();
        double[] hiddenGradients = backPropagationNetwork.getHiddenGradients();

        double derivative;
        double sum = 0.0;
        double delta;

        int numberOfOutputcells = backPropagationNetwork.getNumberOfOutputCells();
        int numberOfHiddenCells = backPropagationNetwork.getNumberOfHiddenCells();
        int numberOfInputCells = backPropagationNetwork.getNumberOfInputCells();

        double learningRate = backPropagationNetwork.getLEARNING_RATE();
        double momentum = backPropagationNetwork.getMomentum();

        double[][] ihPreviousWeightsDelta = backPropagationNetwork.getIhPreviousWeightsDelta();
        double[] hPreviousBiasesDelta = backPropagationNetwork.gethPreviousBiasesDelta();
        double[][] hoPreviousWeightsDelta = backPropagationNetwork.getHoPreviousWeightsDelta();
        double[] oPreviousBiasesDelta = backPropagationNetwork.getoPreviousBiasesDelta();

        // Calculate gradients
        for (int i = 0; i < numberOfOutputcells; i++) {
            derivative = (1 - outputCells[i]) * outputCells[i];
            outputGradients[i] = derivative * (targetValues[i] - outputCells[i]);
        }
        backPropagationNetwork.setOutputGradients(outputGradients);

        for (int i = 0; i < numberOfHiddenCells; i++) {
            derivative = (1 - tempHiddens[i]) * (1 + tempHiddens[i]);
            for (int j = 0; j < backPropagationNetwork.getNumberOfOutputCells(); j++) {
                sum += outputGradients[j] * hoWeights[i][j];
            }
            hiddenGradients[i] = derivative * sum;
        }
        backPropagationNetwork.setHiddenGradients(hiddenGradients);

        // update weights & biases
        for (int i = 0; i < numberOfInputCells; i++)
            for (int j = 0; j < numberOfHiddenCells; j++) {
                delta = learningRate * hiddenGradients[j] * inputCells[i];
                ihWeights[i][j] += delta;
                ihWeights[i][j] += momentum * ihPreviousWeightsDelta[i][j];
                ihPreviousWeightsDelta[i][j] = delta;
            }
        backPropagationNetwork.setIhWeights(ihWeights);
        backPropagationNetwork.setIhPreviousWeightsDelta(ihPreviousWeightsDelta);

        for (int i = 0; i < numberOfHiddenCells; i++){
            delta = learningRate * hiddenGradients[i];
            hiddenBiases[i] += delta;
            hiddenBiases[i] += momentum * hPreviousBiasesDelta[i];
            hPreviousBiasesDelta[i] = delta;
        }
        backPropagationNetwork.sethBiases(hiddenBiases);
        backPropagationNetwork.sethPreviousBiasesDelta(hPreviousBiasesDelta);

        for (int i = 0; i < numberOfHiddenCells; i++)
            for (int j = 0; j < numberOfOutputcells; j++) {
                delta = learningRate * outputGradients[j] * hiddenCells[i];
                hoWeights[i][j] += delta;
                hoWeights[i][j] += momentum * hoPreviousWeightsDelta[i][j];
                hoPreviousWeightsDelta[i][j] = delta;
            }
        backPropagationNetwork.setHoWeights(hoWeights);
        backPropagationNetwork.setHoPreviousWeightsDelta(hoPreviousWeightsDelta);

        for (int i = 0; i < numberOfOutputcells; i++){
            delta = learningRate * outputGradients[i];
            outputBiases[i] += delta;
            outputBiases[i] += momentum * oPreviousBiasesDelta[i];
            oPreviousBiasesDelta[i] = delta;
        }
        backPropagationNetwork.setoBiases(outputBiases);
        backPropagationNetwork.setoPreviousBiasesDelta(oPreviousBiasesDelta);
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
