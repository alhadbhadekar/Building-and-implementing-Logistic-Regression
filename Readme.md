# Logistic Regression from Scratch in Python

This notebook demonstrates the implementation of a Logistic Regression model from scratch in Python using the PIMA Diabetes dataset. The steps include loading the data, standardizing it, training a logistic regression model, and evaluating the model's performance.

## Overview

Logistic regression is a statistical method used for binary classification. It predicts the probability that a given input point belongs to a particular class. In this implementation, we will:

1. Implement logistic regression using gradient descent.
2. Train the model on the PIMA Diabetes dataset.
3. Evaluate its performance using accuracy scores on training and test data.

## Features

- **Learning Rate:** The model uses a learning rate to control the step size during optimization.
- **Gradient Descent:** The optimization algorithm used to minimize the loss function by updating weights and bias.
- **Sigmoid Activation:** The logistic regression function, where the sigmoid function is used to model the decision boundary.

---

## Dependencies

This notebook requires the following Python libraries:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

---

## Steps in the Code

### 1. **Logistic Regression Model**

The `Logistic_Regression` class implements the following methods:

- `__init__(self, learning_rate, no_of_iterations)`: Initializes the learning rate and number of iterations (hyperparameters).
- `fit(self, X, Y)`: Trains the model using the dataset and updates weights and bias via gradient descent.
- `update_weights(self)`: Performs one iteration of gradient descent to update the weights and bias.
- `predict(self, X)`: Makes predictions based on input features using the logistic regression model.

### 2. **Gradient Descent and Sigmoid Function**

The gradient descent algorithm updates the weights (`w`) and bias (`b`) using the following formulas:

- Weight update: \( w = w - \alpha \cdot \frac{\partial J}{\partial w} \)
- Bias update: \( b = b - \alpha \cdot \frac{\partial J}{\partial b} \)

Where:
- \( \alpha \) is the learning rate.
- \( J \) is the cost function (mean squared error).

The sigmoid function is used to calculate predicted probabilities:

\[ \hat{Y} = \frac{1}{1 + e^{-(X \cdot w + b)}} \]

### 3. **PIMA Diabetes Dataset**

The dataset used here contains 8 features (such as glucose, BMI, age) and a binary target indicating whether the person is diabetic or not.

```python
# Loading the dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# Separating features and target labels
features = diabetes_dataset.drop(columns='Outcome', axis=1)
target = diabetes_dataset['Outcome']
```

### 4. **Data Preprocessing**

The features are standardized using `StandardScaler` for better convergence during training.

```python
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)
```

### 5. **Training the Model**

The data is split into training and test sets, and the logistic regression model is trained using the training data.

```python
X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, target, test_size=0.2, random_state=2)
classifier = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
classifier.fit(X_train, Y_train)
```

### 6. **Model Evaluation**

The accuracy of the model is evaluated using the training and test data.

```python
# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
```

### 7. **Making Predictions**

You can use the model to predict whether a new individual is diabetic based on input data.

```python
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)  # example input
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_as_numpy_array)
prediction = classifier.predict(std_data)
```

---

## Output

### Example Output:

```plaintext
Accuracy score of the training data :  0.7735
Accuracy score of the test data :  0.758
The person is diabetic
```

### Conclusion

This notebook shows the end-to-end implementation of a Logistic Regression model from scratch. The model was trained on the PIMA Diabetes dataset, and its performance was evaluated using accuracy scores. The approach demonstrates how to build a machine learning model from the ground up, which helps in understanding the inner workings of algorithms like Logistic Regression.

