import be.kdg.ai.backpropagation.controller.BackPropagationController;
import be.kdg.ai.backpropagation.controller.Controller;
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
public class DemoALetter {
    public static void main(String[] args) {
        BackPropagationNetwork network = new BackPropagationNetwork(4, 2);
        Controller controller = new BackPropagationController(network);

        //Load the A-letter
        List<Double> inputCells = new ArrayList<>();
        Scanner scanner;
        try {
            String filePath = new File("").getAbsolutePath();
            filePath = filePath.concat("//files//A.csv");
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

    public static double[] convertDoubles(List<Double> doubles)
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
