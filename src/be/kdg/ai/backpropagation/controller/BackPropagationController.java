package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.BackPropagationNetwork;
import be.kdg.ai.backpropagation.view.JavaFxView;
import javafx.application.Platform;
import javafx.concurrent.Task;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * The backpropagation algorithm
 */
public class BackPropagationController implements Controller {
    private static final int WAIT_TIME_MILLIS = 10;
    private static final Logger logger = LogManager.getLogger(BackPropagationController.class);
    private final BackPropagationNetwork backPropagationNetwork;
    private final double[] tempHiddens;
    private Thread th;

    public BackPropagationController(BackPropagationNetwork backPropagationNetwork) {
        this.backPropagationNetwork = backPropagationNetwork;
        tempHiddens = new double[backPropagationNetwork.getNumberOfHiddenCells()];
        th = new Thread();
    }

    @Override
    public void stopBackpropagation() {
        if (th.isAlive()) {
            String stoppedMessage = "Backpropagation Stopped";
            th.interrupt();
            JavaFxView.backPropagationStatus(stoppedMessage);
            logger.trace(stoppedMessage);
        }
    }

    @Override
    public void startBackpropagation() {
        String runningMessage = "Running BackPropagation.";
        logger.trace(runningMessage);

        Task task = new Task<Void>() {
            @Override
            public Void call() throws Exception {
                for (int i = 0; i <= backPropagationNetwork.getMAX_EPOCH(); i++) {
                    final int epoch = i;
                    Platform.runLater(() -> {
                        if (epoch == 0) JavaFxView.backPropagationStatus(runningMessage);
                        backPropagationNetwork.setEpoch(epoch);
                        computeOutputs();
                        updateWeights();
                        JavaFxView.changeValues();

                        // Stop if errortreshold is reached
                        double[] errors = backPropagationNetwork.getErrors();
                        double maxError = backPropagationNetwork.getErrorTreshold();
                        boolean allOutputsAreGood = false;
                        for (double error : errors) {
                            if (error < maxError) allOutputsAreGood = true;
                            else {
                                allOutputsAreGood = false;
                                break;
                            }
                        }
                        if (allOutputsAreGood)
                            stopBackpropagation();
                        if (epoch == backPropagationNetwork.getMAX_EPOCH())
                            stopBackpropagation();
                    });
                    Thread.sleep(WAIT_TIME_MILLIS);
                }
                return null;
            }
        };
        th = new Thread(task);
        th.setDaemon(true);
        th.start();
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
        computeErrors();

        //region get neccessary cells and values from the network
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

        double learningRate = backPropagationNetwork.getLearningRate();
        double momentum = backPropagationNetwork.getMomentum();

        double[][] ihPreviousWeightsDelta = backPropagationNetwork.getIhPreviousWeightsDelta();
        double[] hPreviousBiasesDelta = backPropagationNetwork.gethPreviousBiasesDelta();
        double[][] hoPreviousWeightsDelta = backPropagationNetwork.getHoPreviousWeightsDelta();
        double[] oPreviousBiasesDelta = backPropagationNetwork.getoPreviousBiasesDelta();
        //endregion

        //region Calculate gradients
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
        //endregion

        //region update weights & biases
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
        //endregion
    }

    /**
     * Prevents that the updating of the network happens too abrupt
     * @param x the value where the hyperTan is going to be calculated for
     * @return hyperTan
     */
    private static double hyperTanFunction(double x)
    {
        if (x < -45.0)
            return -1.0;
        if (x > 45.0)
            return 1.0;
        return Math.tanh(x);
    }

    /**
     * same as @hyperTanFunction
     * @param x the value where the sigmoid is going to be calculated for
     * @return sigmoid
     */
    private static double sigmoidFunction(double x)
    {
        if (x < -45.0)
            return 0.0;
        if (x > 45.0)
            return 1.0;
        return 1.0/(1.0 + Math.exp(-x));
    }

}
