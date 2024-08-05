def evaluate_model(model, x_test, y_test):
    """Evaluate the model and return accuracy and loss."""
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy
