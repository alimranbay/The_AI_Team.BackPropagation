package be.kdg.ai.backpropagation.view;

import be.kdg.ai.backpropagation.controller.ViewController;
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
 * This is class starts the JavaFX application.
 */
public class JavaFxView extends Application{
    // http://stackoverflow.com/questions/15160410/usage-of-javafx-platform-runlater-and-access-to-ui-from-a-different-thread?lq=1
    private static ViewController viewController;
    private static Label[] inputLabels;
    private static Label[] targetLabels;
    private static Label[] hiddenLabels;
    private static Label[] ihLabels;
    private static Label[] hoLabels;
    private static Label[] outputLabels;

    public void setViewController(ViewController viewController) {
        JavaFxView.viewController = viewController;
    }

    @Override
    public void start(Stage primaryStage) throws Exception{
        // Parent root = FXMLLoader.load(getClass().getResource("sample.fxml"));
        Group root = new Group();
        primaryStage.setTitle("Neurale netwerk");
        GridPane grid = new GridPane();
        Scene scene = new Scene(grid, 1000, 600);
        scene.getStylesheets().add("/be/kdg/ai/backpropagation/view/css/layout.css");

        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(10, 10, 10, 10));
        grid.getStyleClass().add("grid");
        // grid.setStyle("-fx-background-color: cadetblue;");

        HBox hbTop = new HBox(10);
        hbTop.setPadding(new Insets(10, 10, 10, 10));
        hbTop.setAlignment(Pos.CENTER);
        hbTop.setMinWidth(980);

        Button initBtn = new Button("Initialise");
        Button startBtn = new Button("Start BackPropgation");

        hbTop.getChildren().add(initBtn);
        hbTop.getChildren().add(startBtn);

        grid.add(hbTop, 0, 0);
        grid.add(createInputCells(),0,1);
        grid.add(createInputToHidden(),0,2);
        grid.add(createHiddenCells(),0,3);
        grid.add(createHiddenToOutput(),0,4);
        grid.add(createOutputCells(),0,5);
        grid.add(createTargets(),0,6);
        grid.add(createTextField(),0,7);

        initBtn.setOnMouseClicked(event -> initialize());
//TODO:        startBtn.setOnMouseClicked(event -> );

        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void initialize() {
        viewController.initializeNetwork();
        double[] inputValues = viewController.getInputValues();
        for (int i = 0; i < inputValues.length; i++)
            inputLabels[i].setText(inputValues[i] + "");
        double[] targetValues = viewController.getTargetValues();
        for (int i = 0; i < targetValues.length; i++)
            targetLabels[i].setText(targetValues[i] + "");
        double[] hiddenValues = viewController.getHiddenValues();
        for (int i = 0; i < hiddenValues.length; i++)
            hiddenLabels[i].setText(hiddenValues[i] + "");
    }

    private HBox createTargets() {
        HBox hBox = new HBox(200);
        hBox.setAlignment(Pos.CENTER);
        hBox.setMinWidth(980);

        Label label = new Label("---- Targets ----");
        label.getStyleClass().add("label");

        targetLabels = new Label[2];
        targetLabels[0] = new Label("");
        targetLabels[0].getStyleClass().add("label");
        targetLabels[1] = new Label("");
        targetLabels[0].getStyleClass().add("label");

        hBox.getChildren().add(targetLabels[0]);
        hBox.getChildren().add(label);
        hBox.getChildren().add(targetLabels[1]);

        return hBox;
    }

    private HBox createInputCells(){
        final int NUMBER_OF_INPUTS = 3;
        inputLabels = new Label[NUMBER_OF_INPUTS];

        HBox hbMid = new HBox(200);
        hbMid.setAlignment(Pos.CENTER);
        hbMid.setMinWidth(980);

        for (int i = 1; i < NUMBER_OF_INPUTS + 1; i++){
            Label temp = new Label("Input "+ String.valueOf(i));
            temp.setPadding(new Insets(20, 20, 20, 20));
            hbMid.getChildren().add(temp);
            temp.getStyleClass().add("input");

            inputLabels[i-1] = temp;
        }

        return hbMid;
    }

    private HBox createHiddenCells(){
        final int NUMBER_OF_HIDDEN_CELLS = 4;
        HBox hbMid2 = new HBox(10);
        hbMid2.setAlignment(Pos.CENTER);
        hbMid2.setMinWidth(980);

        hiddenLabels = new Label[NUMBER_OF_HIDDEN_CELLS];

        for (int i = 1; i < NUMBER_OF_HIDDEN_CELLS + 1; i++){
            Label temp = new Label("Hidden Cell "+ String.valueOf(i));
            temp.setPadding(new Insets(20, 40, 20, 40));
            hbMid2.getChildren().add(temp);
            temp.getStyleClass().add("input");
            hiddenLabels[i-1] = temp;
        }

        return hbMid2;
    }

    private HBox createOutputCells(){
        HBox hbMid3 = new HBox(230);
        hbMid3.setAlignment(Pos.CENTER);
        hbMid3.setMinWidth(980);

        for (int i = 1; i < 3; i++){
            Label temp = new Label("Output Cell "+ String.valueOf(i));
            temp.setPadding(new Insets(20, 40, 20, 40));
            hbMid3.getChildren().add(temp);
            temp.getStyleClass().add("input");
        }

        return hbMid3;
    }

    private HBox createInputToHidden(){
        HBox hbMid = new HBox(50);
        hbMid.setAlignment(Pos.CENTER);
        hbMid.setMinWidth(980);

        HBox h1 = new HBox(15);
        for (int i = 1; i < 5; i++){
            Label temp = new Label("IH1"+ String.valueOf(i));
            temp.setPadding(new Insets(20, 10, 20, 10));
            h1.getChildren().add(temp);
            temp.getStyleClass().add("hidden");
        }

        HBox h2 = new HBox(15);
        for (int i = 1; i < 5; i++){
            Label temp = new Label("IH2"+ String.valueOf(i));
            temp.setPadding(new Insets(20, 10, 20, 10));
            h2.getChildren().add(temp);
            temp.getStyleClass().add("hidden");
        }

        HBox h3 = new HBox(15);
        for (int i = 1; i < 5; i++){
            Label temp = new Label("IH3"+ String.valueOf(i));
            temp.setPadding(new Insets(20, 10, 20, 10));
            h3.getChildren().add(temp);
            temp.getStyleClass().add("hidden");
        }

        hbMid.getChildren().add(h1);
        hbMid.getChildren().add(h2);
        hbMid.getChildren().add(h3);

        return hbMid;

    }

    private HBox createHiddenToOutput(){
        HBox hbMid = new HBox(10);
        hbMid.setAlignment(Pos.CENTER);
        hbMid.setMinWidth(980);

        HBox h1 = new HBox(45);
        for (int i = 1; i < 3; i++){
            Label temp = new Label("HO1"+ String.valueOf(i));
            temp.setPadding(new Insets(20, 10, 20, 10));
            h1.getChildren().add(temp);
            temp.getStyleClass().add("hidden");
        }

        HBox h2 = new HBox(45);
        for (int i = 1; i < 3; i++){
            Label temp = new Label("HO2"+ String.valueOf(i));
            temp.setPadding(new Insets(20, 10, 20, 10));
            h2.getChildren().add(temp);
            temp.getStyleClass().add("hidden");
        }

        HBox h3 = new HBox(45);
        for (int i = 1; i < 3; i++){
            Label temp = new Label("HO3"+ String.valueOf(i));
            temp.setPadding(new Insets(20, 10, 20, 10));
            h3.getChildren().add(temp);
            temp.getStyleClass().add("hidden");
        }

        HBox h4 = new HBox(45);
        for (int i = 1; i < 3; i++){
            Label temp = new Label("HO4"+ String.valueOf(i));
            temp.setPadding(new Insets(20, 10, 20, 10));
            h4.getChildren().add(temp);
            temp.getStyleClass().add("hidden");
        }
        hbMid.getChildren().add(h1);
        hbMid.getChildren().add(h2);
        hbMid.getChildren().add(h3);
        hbMid.getChildren().add(h4);

        return hbMid;

    }

    private HBox createTextField(){
        HBox hbMid = new HBox(20);
        hbMid.setAlignment(Pos.CENTER);
        hbMid.setMinWidth(980);

        Label labelbt = new Label("Learning Rate");
        labelbt.setPadding(new Insets(10, 10, 10, 10));
        Label labelbt2 = new Label("Error Threshold");
        labelbt2.setPadding(new Insets(10, 10, 10, 10));

        TextField forLabel1 = new TextField();
        hbMid.getChildren().addAll(labelbt, forLabel1);
        TextField forLabel2 = new TextField();
        hbMid.getChildren().addAll(labelbt2, forLabel2);

        return hbMid;
    }
}