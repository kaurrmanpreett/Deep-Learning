from modules.imports import *
from modules.data_preparation import load_and_prepare_data
from modules.model_building import create_model
from modules.model_training import train_model
from modules.model_evaluation import evaluate_model
from modules.plotting import plot_learning_curve, plot_lr_vs_loss
from modules.best_learning_rate import find_best_learning_rate
from modules.making_prediction import make_predictions
from modules.evaluation_metrics import calculate_metrics

def main():
    # File path
    filepath = 'C:\\Users\\kaurr\\OneDrive\\Desktop\\BISI\\2208\\Deep Learning\\data\\employee_attrition.csv'
    
    # Data Preparation
    x_train, x_test, y_train, y_test = load_and_prepare_data(filepath)
    
    # Create and train model
    model = create_model(learning_rate=0.0009, extra_layers=1, neurons=1)
    history = train_model(model, x_train, y_train, epochs=50)
    
    # Evaluate model
    loss, accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    
    # Plot training curves
    plot_learning_curve(history)
    
    # Find best learning rate
    find_best_learning_rate(x_train, y_train)
    
    # Make predictions
    y_preds = make_predictions(model, x_test)
    
    # Calculate and print metrics
    calculate_metrics(y_test, y_preds)
    
if __name__ == "__main__":
    main()
