import tensorflow as tf

def make_predictions(model, x_test):
    """Make predictions and convert to binary format."""
    y_preds = tf.round(model.predict(x_test))
    return y_preds
