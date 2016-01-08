package be.kdg.ai.backpropagation.model;

/**
 * This exception will be thrown when an error occurs during initialisation of the Backpropagation algorithm.
 */
public class InitialisationException extends Exception {
    private String errorMessage;

    public InitialisationException(String errorMessage){
        this.errorMessage = errorMessage;
    }

    @Override
    public String toString(){
        return errorMessage;
    }
}
