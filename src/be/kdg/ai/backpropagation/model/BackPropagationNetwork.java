package be.kdg.ai.backpropagation.model;

import java.util.Random;

/**
 * The backpropagation network that's going to be used in the backpropagation algorithm
 */
public class BackPropagationNetwork {

    //region Fields

    private boolean isInitialized;
    private int numberOfInputCells;
    private final int numberOfHiddenCells;
    private final int numberOfOutputCells;

    private double[] inputCells = new double[]{1.0, -2.0, 3.0};                // Hard zetten op 1.0, -2.0 en 3.0

    private double[][] ihWeights;               // input-to-hidden

    private double[] hBiases;
    private double[] hiddenCells;

    private double[][] hoWeights;               // hidden-to-output

    private double[] oBiases;
    private double[] outputCells;

    private double[] targets;

    private int epoch; // Aantal leeriteraties
    private static final int MAX_EPOCH = 100;

    private double[] errors; // Verschil tussen waarden van output cellen en target values na elke iteratie
    private double errorTreshold; // VOORLOPIG. If error < errorTreshold dan kan het leren stoppen

    private double momentum;
    private double learningRate; // VOORLOPIG

    private double[] outputGradients;
    private double[] hiddenGradients;

    private double[][] ihPreviousWeightsDelta;
    private double[] hPreviousBiasesDelta;
    private double[][] hoPreviousWeightsDelta;
    private double[] oPreviousBiasesDelta;

    //endregion

    /**
     *
     * @param numberOfHiddenCells .
     * @param numberOfOutputCells .
     */
    public BackPropagationNetwork(int numberOfHiddenCells, int numberOfOutputCells, double errorTreshold, double learningRate, double momentum){
        isInitialized = false;
        this.numberOfHiddenCells = numberOfHiddenCells;
        this.numberOfOutputCells = numberOfOutputCells;
        this.errorTreshold = errorTreshold;
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    /**
     * Initializes the backpropagation network
     * @throws InitialisationException
     */
    public void initialize() throws InitialisationException{
        if(inputCells.length <= 0) throw new InitialisationException("Number of inputcells cannot be less than 1.");
        if(numberOfHiddenCells <= 0) throw new InitialisationException("Number of hiddencells cannot be less than 1.");
        if(numberOfOutputCells <= 0) throw new InitialisationException("Number of outputcells cannot be less than 1.");
//        if(lRate <= 0) throw  new InitialisationException("Learning rate cannot be zero or less.");

        Random random = new Random();

        hiddenCells = new double[numberOfHiddenCells];
        outputCells = new double[numberOfOutputCells];

        numberOfInputCells = inputCells.length;

        ihWeights = new double[numberOfInputCells][numberOfHiddenCells];
        hBiases = new double[numberOfHiddenCells];
        hoWeights = new double[numberOfHiddenCells][numberOfOutputCells];
        oBiases = new double[numberOfOutputCells];

        errors = new double[numberOfOutputCells];

        targets = new double[numberOfOutputCells];

        hiddenGradients = new double[numberOfHiddenCells];
        outputGradients = new double[numberOfOutputCells];

        ihPreviousWeightsDelta = new double[numberOfInputCells][numberOfHiddenCells];
        hoPreviousWeightsDelta = new double[numberOfHiddenCells][numberOfOutputCells];
        hPreviousBiasesDelta = new double[numberOfHiddenCells];
        oPreviousBiasesDelta = new double[numberOfOutputCells];

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

        isInitialized = true;
    }

    public boolean isInitialized() { return isInitialized; }

    public double[] getInputCells() {
        return inputCells;
    }

    public double[] getTargets() {
        return targets;
    }

    public double[] getHiddenCells() {
        return hiddenCells;
    }

    public double[][] getIhWeights() {
        return ihWeights;
    }

    public double[][] getHoWeights() {
        return hoWeights;
    }

    public void setOutputCells(double[] outputCells) {
        this.outputCells = outputCells;
    }

    public void setErrors(double[] errors) {
        this.errors = errors;
    }

    public int getNumberOfInputCells() {
        return numberOfInputCells;
    }

    public int getNumberOfHiddenCells() {
        return numberOfHiddenCells;
    }

    public int getNumberOfOutputCells() {
        return numberOfOutputCells;
    }

    public void setHiddenCell(int index, double hiddenCell) {
        hiddenCells[index] = hiddenCell;
    }

    public double[] getOutputCells() {
        return outputCells;
    }

    public int getEpoch() {
        return epoch;
    }

    public double[] getErrors() {
        return errors;
    }

    public double getMomentum() {
        return momentum;
    }

    public double[] gethBiases() {
        return hBiases;
    }

    public double[] getoBiases() {
        return oBiases;
    }

    public double[] getHiddenGradients() {
        return hiddenGradients;
    }

    public double[] getOutputGradients() {
        return outputGradients;
    }

    public void setOutputGradients(double[] outputGradients) {
        this.outputGradients = outputGradients;
    }

    public void setHiddenGradients(double[] hiddenGradients) {
        this.hiddenGradients = hiddenGradients;
    }

    public void sethBiases(double[] hBiases) {
        this.hBiases = hBiases;
    }

    public void setIhWeights(double[][] ihWeights) {
        this.ihWeights = ihWeights;
    }

    public void setHoWeights(double[][] hoWeights) {
        this.hoWeights = hoWeights;
    }

    public void setoBiases(double[] oBiases) {
        this.oBiases = oBiases;
    }

    public double[] getoPreviousBiasesDelta() {
        return oPreviousBiasesDelta;
    }

    public void setoPreviousBiasesDelta(double[] oPreviousBiasesDelta) {
        this.oPreviousBiasesDelta = oPreviousBiasesDelta;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double[][] getIhPreviousWeightsDelta() {
        return ihPreviousWeightsDelta;
    }

    public void setIhPreviousWeightsDelta(double[][] ihPreviousWeightsDelta) {
        this.ihPreviousWeightsDelta = ihPreviousWeightsDelta;
    }

    public double[] gethPreviousBiasesDelta() {
        return hPreviousBiasesDelta;
    }

    public void sethPreviousBiasesDelta(double[] hPreviousBiasesDelta) {
        this.hPreviousBiasesDelta = hPreviousBiasesDelta;
    }

    public double[][] getHoPreviousWeightsDelta() {
        return hoPreviousWeightsDelta;
    }

    public void setHoPreviousWeightsDelta(double[][] hoPreviousWeightsDelta) {
        this.hoPreviousWeightsDelta = hoPreviousWeightsDelta;
    }

    public double getErrorTreshold() {
        return errorTreshold;
    }

    public int getMAX_EPOCH() {
        return MAX_EPOCH;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setErrorTreshold(double errorTreshold) {
        this.errorTreshold = errorTreshold;
    }

    public void setInputCells(double[] inputCells) {
        this.inputCells = inputCells;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }
}
