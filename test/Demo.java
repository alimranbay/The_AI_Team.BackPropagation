import be.kdg.ai.backpropagation.controller.Controller;
import be.kdg.ai.backpropagation.controller.ViewController;
import be.kdg.ai.backpropagation.model.BackPropagationNetwork;
import be.kdg.ai.backpropagation.model.NeuralNetwork;
import be.kdg.ai.backpropagation.view.JavaFxView;

/**
 * This class starts the app.
 */
public class Demo {
    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new BackPropagationNetwork(4, 2);
        Controller controller = new Controller(neuralNetwork);
//        controller.startNeuralNetwork();

        ViewController viewController = new ViewController(neuralNetwork, controller);
        JavaFxView javaFxView = new JavaFxView();
        javaFxView.setViewController(viewController);

        JavaFxView.launch(JavaFxView.class, args);
    }
}
