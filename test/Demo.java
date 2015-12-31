import be.kdg.ai.backpropagation.controller.BackPropagationController;
import be.kdg.ai.backpropagation.controller.Controller;
import be.kdg.ai.backpropagation.controller.ViewController;
import be.kdg.ai.backpropagation.model.BackPropagationNetwork;
import be.kdg.ai.backpropagation.view.JavaFxView;

/**
 * This class starts the app.
 */
public class Demo {
    public static void main(String[] args) {
        BackPropagationNetwork network = new BackPropagationNetwork(4, 2);
        Controller controller = new BackPropagationController(network);
//        controller.startNeuralNetwork();

        ViewController viewController = new ViewController(network, controller);
        JavaFxView javaFxView = new JavaFxView();
        javaFxView.setViewController(viewController);

        JavaFxView.launch(JavaFxView.class, args);
    }
}
