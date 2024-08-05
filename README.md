# Deep-learning
How a deep neural network learns
A model learns by updating and improving its weights and biases every epoch (when we call the fit() function).

It does so by comparing the patterns its learned between the data and labels to the actual labels.

If the current patterns (weight matrices and bias values) don't result in a desirable decrease in the loss function (higher loss means worse predictions), the optimizer tries to steer the model to update its patterns in the right way (using the real labels as a reference).

This process of using the real labels as a reference to improve the model's predictions is called backpropagation.

In other words, data and labels pass through a model (forward pass) and it attempts to learn the relationship between the data and labels.

If this learned relationship isn't close to the actual relationship or it could be improved, the model does so by going back through itself (backward pass) and tweaking its weights and bias values to better represent the data.

Installation:
To use this project, clone the repository and install the required packages:
      git clone https://github.com/kaurrmanpreett/Deep-Learning.git

      ```sh
      pip install -r requirements.txt

Usage:
1. Place your time series data files in the data/ directory.
2. Modify main.py to load and analyze your data as needed.
3. Run the main script:
      python main.py

Modules
1. best_learning_rate.py: This module contains functions to determine the best learning rate for training models.

2. find_best_learning_rate: Finds the optimal learning rate based on the provided data.

3. data_preparation_.py: This module contains functions to prepare and clean the data.
prepare_data: Prepares the dataset for analysis.

4. evaluation_metrics.py: This module contains functions to calculate various evaluation metrics for model performance.
calculate_metrics(predictions, targets): Calculates evaluation metrics for the given predictions and targets.

5. imports.py: This module contains import statements and configurations required for the project.

6. making_prediction.py: This module contains functions to make predictions using the trained model.
make_prediction: Uses the trained model to make predictions on the provided data.

7. model_building.py: This module contains functions to build machine learning models.
build_model: Builds a machine learning model based on the provided data.

8. model_training.py: This module contains functions to train the models.

9. model_evaluation.py: This module contains functions to evaluate the performance of the models.
evaluate_model: Evaluates the performance of the model using the provided data.
train_model: Trains the model using the provided data.

10. plotting.py: This module contains functions to plot graphs and visualizations.
plot_data: Plots the given data.

License
This project is licensed under the Apache License 2.0.





 
