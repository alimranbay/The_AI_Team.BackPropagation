import be.kdg.ai.backpropagation.controller.BackPropagationController;
import be.kdg.ai.backpropagation.controller.Controller;
import be.kdg.ai.backpropagation.controller.CsvWriter;
import be.kdg.ai.backpropagation.controller.ViewController;
import be.kdg.ai.backpropagation.model.BackPropagationNetwork;
import be.kdg.ai.backpropagation.view.JavaFxView;

/**
 * This class starts the app.
 */
class Demo {
    public static void main(String[] args) {
        //default values
        int numberOfHiddenCells =4, numberOfOutputCells =2;
        double errorThreshold=0.0001, learningRate=0.5, momentum=0.1;

        BackPropagationNetwork network = new BackPropagationNetwork(numberOfHiddenCells, numberOfOutputCells, errorThreshold, learningRate, momentum);

        CsvWriter csvWriter = new CsvWriter(network);
        csvWriter.setFileName("demo.csv");
        Controller controller = new BackPropagationController(network, csvWriter);

        ViewController viewController = new ViewController(network);
        JavaFxView javaFxView = new JavaFxView();
        javaFxView.setViewController(viewController);

        JavaFxView.setController(controller);
        JavaFxView.launch(JavaFxView.class, args);
    }
}
