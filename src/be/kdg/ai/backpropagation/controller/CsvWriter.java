package be.kdg.ai.backpropagation.controller;

import be.kdg.ai.backpropagation.model.BackPropagationNetwork;
import com.opencsv.CSVWriter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This class writes the backPropagationNetwork to a csv file
 */
public class CsvWriter {
    private BackPropagationNetwork backPropagationNetwork;
    private static final Logger logger = LogManager.getLogger(CsvWriter.class);
    private String fileName;


    public CsvWriter(BackPropagationNetwork backPropagationNetwork, String fileName) {
        this.backPropagationNetwork = backPropagationNetwork;
        this.fileName = fileName;
    }

    public CsvWriter(BackPropagationNetwork backPropagationNetwork) {
        this.backPropagationNetwork = backPropagationNetwork;
    }

    public void saveCurrentNetworkAsCsv() {
        try
        {
            CSVWriter csvWriter = new CSVWriter(new FileWriter(fileName));
            final int INPUT_POSITION = 0;
            final int IH_POSITION = 1;
            final int HIDDEN_POSITION = 2;
            final int HO_POSITION = 3;
            final int OUTPUT_POSITION = 4;
            final int TARGET_POSITION = 5;

            List<String[]> csvRows = new ArrayList<>();

            for (int i = 0; i < backPropagationNetwork.getNumberOfInputCells(); i++) {
                String inputCell = String.valueOf(backPropagationNetwork.getInputCells()[i]);
                try {
                    String[] row = csvRows.get(i);
                    row[INPUT_POSITION] = inputCell;
                    csvRows.set(i, row);
                }
                catch (IndexOutOfBoundsException e) {
                    String[] row = {inputCell, "", "", "", "", ""};
                    csvRows.add(row);
                }
            }

            for (int i = 0; i < backPropagationNetwork.getNumberOfInputCells(); i++) {
                for (int j = 0; j < backPropagationNetwork.getNumberOfHiddenCells(); j++) {
                    String ihWeight = String.valueOf(backPropagationNetwork.getIhWeights()[i][j]);
                    try {
                        int oneDimensionalIndex = (i * backPropagationNetwork.getNumberOfHiddenCells()) + j;
                        String[] row = csvRows.get(oneDimensionalIndex);
                        row[IH_POSITION] = ihWeight;
                        csvRows.set(oneDimensionalIndex, row);
                    }
                    catch (IndexOutOfBoundsException e) {
                        String[] row = {"", ihWeight, "", "", "", ""};
                        csvRows.add(row);
                    }
                }
            }

            for (int i = 0; i < backPropagationNetwork.getNumberOfHiddenCells(); i++) {
                String hiddenCell = String.valueOf(backPropagationNetwork.getHiddenCells()[i]);
                try {
                    String[] row = csvRows.get(i);
                    row[HIDDEN_POSITION] = hiddenCell;
                    csvRows.set(i, row);
                }
                catch (IndexOutOfBoundsException e) {
                    String[] row = {"", "", hiddenCell, "", "", ""};
                    csvRows.add(row);
                }
            }

            for (int i = 0; i < backPropagationNetwork.getNumberOfHiddenCells(); i++) {
                for (int j = 0; j < backPropagationNetwork.getNumberOfOutputCells(); j++) {
                    String hoWeight = String.valueOf(backPropagationNetwork.getHoWeights()[i][j]);
                    try {
                        int oneDimensionalIndex = (i * backPropagationNetwork.getNumberOfOutputCells()) + j;
                        String[] row = csvRows.get(oneDimensionalIndex);
                        row[HO_POSITION] = hoWeight;
                        csvRows.set(oneDimensionalIndex, row);
                    }
                    catch (IndexOutOfBoundsException e) {
                        String[] row = {"", "", "", hoWeight, "", ""};
                        csvRows.add(row);
                    }
                }
            }

            for (int i = 0; i < backPropagationNetwork.getNumberOfOutputCells(); i++) {
                String outputCell = String.valueOf(backPropagationNetwork.getOutputCells()[i]);
                String target = String.valueOf(backPropagationNetwork.getTargets()[i]);
                try {
                    String[] row = csvRows.get(i);
                    row[OUTPUT_POSITION] = outputCell;
                    row[TARGET_POSITION] = target;
                    csvRows.set(i, row);
                }
                catch (IndexOutOfBoundsException e) {
                    String[] row = {"", "", "", "", outputCell, target};
                    csvRows.add(row);
                }
            }
            String[] header = {"inputCell", "ihWeights", "hiddenCell", "hoWeight", "outputCell", "target"};

            csvRows.add(0, header);

            csvWriter.writeAll(csvRows);
            csvWriter.flush();
            csvWriter.close();
        }
        catch (IOException e)
        {
            logger.error(e.getMessage());
        }
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }
}
