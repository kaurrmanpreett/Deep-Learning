def train_model(model, x_train, y_train, epochs=50, verbose=0):
    """Train the neural network model."""
    history = model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
    return history
