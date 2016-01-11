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
        BackPropagationNetwork network = new BackPropagationNetwork(4, 2);
        CsvWriter csvWriter = new CsvWriter(network, "demo.csv");
        Controller controller = new BackPropagationController(network, csvWriter);

        ViewController viewController = new ViewController(network);
        JavaFxView javaFxView = new JavaFxView();
        javaFxView.setViewController(viewController);

        JavaFxView.setController(controller);
        JavaFxView.launch(JavaFxView.class, args);
    }
}
