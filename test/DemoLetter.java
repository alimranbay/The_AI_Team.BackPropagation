import be.kdg.ai.backpropagation.controller.BackPropagationController;
import be.kdg.ai.backpropagation.controller.Controller;
import be.kdg.ai.backpropagation.controller.CsvWriter;
import be.kdg.ai.backpropagation.controller.ViewController;
import be.kdg.ai.backpropagation.model.BackPropagationNetwork;
import be.kdg.ai.backpropagation.view.JavaFxView;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

/**
 * This class starts the app.
 */
class DemoLetter {
    public static void main(String[] args) {
        //default values
        int numberOfHiddenCells =4, numberOfOutputCells =2;
        double errorThreshold=0.0001, learningRate=0.5, momentum=0.1;

        BackPropagationNetwork network = new BackPropagationNetwork(numberOfHiddenCells, numberOfOutputCells, errorThreshold, learningRate, momentum);

        CsvWriter csvWriter = new CsvWriter(network);
        Controller controller = new BackPropagationController(network, csvWriter);

        //Load the character
        List<Double> inputCells = new ArrayList<>();
        Scanner scanner;
        try {
            String filePath = new File("").getAbsolutePath();
            // choose A.csv or B.csv
            String character = "B";
            csvWriter.setFileName(character + "_output.csv");
            filePath = filePath.concat("//files//" + character + ".csv");
            scanner = new Scanner(new File(filePath));
            scanner.useDelimiter(",");
            while(scanner.hasNext()){
                inputCells.add(((double) scanner.nextInt()));
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        network.setInputCells(convertDoubles(inputCells));

        ViewController viewController = new ViewController(network);
        JavaFxView javaFxView = new JavaFxView();
        javaFxView.setViewController(viewController);

        JavaFxView.setController(controller);
        JavaFxView.launch(JavaFxView.class, args);
    }

    private static double[] convertDoubles(List<Double> doubles)
    {
        double[] ret = new double[doubles.size()];
        Iterator<Double> iterator = doubles.iterator();
        for (int i = 0; i < ret.length; i++)
        {
            ret[i] = iterator.next();
        }
        return ret;
    }
}
