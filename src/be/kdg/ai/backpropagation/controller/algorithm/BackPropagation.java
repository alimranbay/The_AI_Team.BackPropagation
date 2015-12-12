package be.kdg.ai.backpropagation.controller.algorithm;

import java.util.Random;

/**
 * A backpropagetion algorithm.
 */
public class BackPropagation implements NeuralNetwork {

    //region Fields

    private int numberOfInputCells;
    private int numberOfHiddenCells;
    private int numberOfOutputCells;

    private double[] inputCells;                // Hard zetten op 1.0, -2.0 en 3.0

    private double[][] ihWeights;               // input-to-hidden

    private double[] hBiases;
    private double[] hiddenCells;

    private double[][] hoWeights;               // hidden-to-output

    private double[] oBiases;
    private double[] outputCells;

    private double[] targets;

    private int epoch; // Aantal leeriteraties
    private static final int MAX_EPOCH = 10_000;

    private double[] errors; // Verschil tussen waarden van output cellen en target values na elke iteratie
    private static final double ERROR_TRESHOLD = 0.0001; // VOORLOPIG If error < ERROR_TRESHOLD dan kan het leren stoppen

    private double momentum;
    private static final double LEARNING_RATE = 0.5; // VOORLOPIG

    //endregion

    /**
     *
     * @param numberOfHiddenCells .
     * @param numberOfOutputCells .
     */
    public BackPropagation(int numberOfHiddenCells, int numberOfOutputCells){
        this.numberOfHiddenCells = numberOfHiddenCells;
        this.numberOfOutputCells = numberOfOutputCells;

        initialize();
    }

    @Override
    public void initialize(){
        Random random = new Random();

        hiddenCells = new double[numberOfHiddenCells];
        outputCells = new double[numberOfOutputCells];

        inputCells = new double[]{1.0, -2.0, 3.0};
        numberOfInputCells = inputCells.length;

        ihWeights = new double[numberOfInputCells][numberOfHiddenCells];
        hBiases = new double[numberOfHiddenCells];
        hoWeights = new double[numberOfHiddenCells][numberOfOutputCells];
        oBiases = new double[numberOfOutputCells];

        errors = new double[numberOfOutputCells];

        targets = new double[numberOfOutputCells];

        for (int i = 0; i < numberOfOutputCells; i++)
            targets[i] = random.nextDouble();
        for (int i = 0; i < numberOfHiddenCells; i++)
            hBiases[i] = random.nextDouble() / 10;
        for (int i = 0; i < numberOfOutputCells; i++)
            oBiases[i] = random.nextDouble() / 10;

        for (int i = 0; i < numberOfInputCells; i++) {
            for (int j = 0; j < numberOfHiddenCells; j++)
                ihWeights[i][j] = random.nextDouble() / 100;
        }

        for (int i = 0; i < numberOfHiddenCells; i++) {
            for (int j = 0; j < numberOfOutputCells; j++)
                hoWeights[i][j] = random.nextDouble() / 100;
        }
    }

    @Override
    public double[] computeOutputs(){
        // initialize hidden cells
        for (int i = 0; i < numberOfHiddenCells; i++)
            hiddenCells[i] = 0;

        for (int i = 0; i < numberOfHiddenCells; i++) {
            for (int j = 0; j < numberOfInputCells; j++)
                hiddenCells[i] += (inputCells[j] * ihWeights[j][i]);
            hiddenCells[i] += hBiases[i];
        }

        double[] tempHiddens = new double[numberOfHiddenCells];
        for (int i = 0; i < numberOfHiddenCells; i++)
            tempHiddens[i] = hyperTanFunction(hiddenCells[i]);

        for (int i = 0; i < numberOfOutputCells; i++) {
            for (int j = 0; j < numberOfHiddenCells; j++)
                outputCells[i] += (tempHiddens[j] * hoWeights[j][i]);
            outputCells[i] += oBiases[i];
        }

        for (int i = 0; i < numberOfOutputCells; i++)
            outputCells[i] = sigmoidFunction(outputCells[i]);

        return outputCells;
    }

    @Override
    public double[] computeErrors() {
        for (int i = 0; i < numberOfOutputCells; i++)
            errors[i] = Math.abs(targets[i] - outputCells[i]);
        return errors;
    }

    public static double hyperTanFunction(double x)
    {
        if (x < -45.0)
            return -1.0;
        if (x > 45.0)
            return 1.0;
        return Math.tanh(x);
    }

    public static double sigmoidFunction(double x)
    {
        if (x < -45.0)
            return 0.0;
        if (x > 45.0)
            return 1.0;
        return 1.0/(1.0 + Math.exp(-x));
    }

    @Override
    public double[] getInputCells() {
        return inputCells;
    }

    @Override
    public double[] getTargets() {
        return targets;
    }
}
