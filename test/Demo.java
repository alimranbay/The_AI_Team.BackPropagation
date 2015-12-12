import be.kdg.ai.backpropagation.controller.Controller;
import be.kdg.ai.backpropagation.controller.ViewController;
import be.kdg.ai.backpropagation.controller.algorithm.BackPropagation;
import be.kdg.ai.backpropagation.controller.algorithm.NeuralNetwork;
import be.kdg.ai.backpropagation.view.JavaFxView;
import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;

/**
 * This class starts the app.
 */
public class Demo {
    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new BackPropagation(4, 2);
        Controller controller = new Controller(neuralNetwork);
//        controller.startNeuralNetwork();

        ViewController viewController = new ViewController(neuralNetwork);
        JavaFxView javaFxView = new JavaFxView();
        javaFxView.setViewController(viewController);

        JavaFxView.launch(JavaFxView.class, args);
    }
}
