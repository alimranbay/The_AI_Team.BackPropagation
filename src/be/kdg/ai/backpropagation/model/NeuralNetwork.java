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
}
