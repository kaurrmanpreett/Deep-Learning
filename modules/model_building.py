import tensorflow as tf

def create_model(learning_rate=0.001, extra_layers=0, neurons=1):
    """Create and compile the neural network model."""
    # Create model
    model = tf.keras.Sequential()
    
    # Add layers
    for _ in range(extra_layers):
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    return model
