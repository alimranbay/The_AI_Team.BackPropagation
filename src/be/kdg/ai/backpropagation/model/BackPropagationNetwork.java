package be.kdg.ai.backpropagation.model;

import java.util.Random;

/**
 * A backpropagetion algorithm.
 */
public class BackPropagationNetwork implements NeuralNetwork {

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
    public BackPropagationNetwork(int numberOfHiddenCells, int numberOfOutputCells){
        this.numberOfHiddenCells = numberOfHiddenCells;
        this.numberOfOutputCells = numberOfOutputCells;

//        initialize();
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
    public double[] getInputCells() {
        return inputCells;
    }

    @Override
    public double[] getTargets() {
        return targets;
    }

    @Override
    public double[] getHiddenCells() {
        return hiddenCells;
    }

    @Override
    public double[][] getIhWeights() {
        return ihWeights;
    }

    @Override
    public double[][] getHoWeights() {
        return hoWeights;
    }

    @Override
    public void setOutputCells(double[] outputCells) {
        this.outputCells = outputCells;
    }

    @Override
    public void setErrors(double[] errors) {
        this.errors = errors;
    }

    @Override
    public int getNumberOfInputCells() {
        return numberOfInputCells;
    }

    @Override
    public int getNumberOfHiddenCells() {
        return numberOfHiddenCells;
    }

    @Override
    public int getNumberOfOutputCells() {
        return numberOfOutputCells;
    }

    @Override
    public void setHiddenCell(int index, double hiddelCell) {
        hiddenCells[index] = hiddelCell;
    }

    @Override
    public double[] getOutputCells() {
        return outputCells;
    }

    @Override
    public int getEpoch() {
        return epoch;
    }

    @Override
    public double[] getErrors() {
        return errors;
    }

    @Override
    public double getMomentum() {
        return momentum;
    }

    @Override
    public double[] gethBiases() {
        return hBiases;
    }

    @Override
    public double[] getoBiases() {
        return oBiases;
    }
}
