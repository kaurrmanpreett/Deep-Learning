from .model_building import create_model
import tensorflow as tf
from .plotting import plot_lr_vs_loss


def find_best_learning_rate(x_train, y_train):
    """Find the best learning rate using a scheduler."""
    def lr_schedule(epoch):
        return 0.001 * 0.9**(epoch/3)

    model = create_model(learning_rate=0.001)
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    history = model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[lr_scheduler])
    plot_lr_vs_loss(history)
    
    return history
