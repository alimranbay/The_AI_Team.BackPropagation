package be.kdg.ai.backpropagation.controller.algorithm;

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

    public BackPropagation(int numberOfHiddenCells, int numberOfOutputCells){
        this.numberOfHiddenCells = numberOfHiddenCells;
        this.numberOfOutputCells = numberOfOutputCells;

        initialize();
    }

    private void initialize(){
        hiddenCells = new double[numberOfHiddenCells];
        outputCells = new double[numberOfOutputCells];

        inputCells = new double[]{1.0, -2.0, 3.0};
        numberOfInputCells = inputCells.length;

        ihWeights = new double[numberOfInputCells][numberOfHiddenCells];
        hBiases = new double[numberOfHiddenCells];
        hoWeights = new double[numberOfHiddenCells][numberOfOutputCells];
        oBiases = new double[numberOfOutputCells];
        targets = new double[numberOfOutputCells];
    }

    public double[] computeOutputs(){
        // initialize hidden cells
        for (int i = 0; i < numberOfHiddenCells; i++) {
            hiddenCells[i] = 0;
        }


    }
}
