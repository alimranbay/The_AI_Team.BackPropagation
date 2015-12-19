package be.kdg.ai.backpropagation.model;

/**
 * An interface that's going to be implemented by a neural network algorithm.
 */
public interface NeuralNetwork {

    void initialize();

    double[] getInputCells();

    double[] getTargets();

    double[] getHiddenCells();

    double[][] getIhWeights();


    double[][] getHoWeights();

    void setOutputCells(double[] outputCells);

    void setErrors(double[] errors);

    int getNumberOfInputCells();

    int getNumberOfHiddenCells();

    int getNumberOfOutputCells();

    void setHiddenCell(int index, double hiddelCell);

    double[] getOutputCells();

    int getEpoch();

    double[] getErrors();

    double getMomentum();

    double[] gethBiases();

    double[] getoBiases();

    double[] getHiddenGradients();

    double[] getOutputGradients();

    void setOutputGradients(double[] outputGradients);

    void setHiddenGradients(double[] hiddenGradients);

    void sethBiases(double[] hBiases);

    void setIhWeights(double[][] ihWeights);

    void setHoWeights(double[][] hoWeights);

    void setoBiases(double[] oBiases);

    double[] getoPreviousBiasesDelta();

    void setoPreviousBiasesDelta(double[] oBiasesPreviousWeightsDelta);

    double getLEARNING_RATE();

    double[][] getIhPreviousWeightsDelta();

    void setIhPreviousWeightsDelta(double[][] ihPreviousWeightsDelta);

    double[] gethPreviousBiasesDelta();

    void sethPreviousBiasesDelta(double[] hBiasesPreviousWeightsDelta);

    double[][] getHoPreviousWeightsDelta();

    void setHoPreviousWeightsDelta(double[][] hoPreviousWeightsDelta);

    double getERROR_TRESHOLD();

    int getMAX_EPOCH();

    void setLEARNING_RATE(double LEARNING_RATE);

    void setMAX_EPOCH(int MAX_EPOCH);

    void setERROR_TRESHOLD(double ERROR_TRESHOLD);
}
