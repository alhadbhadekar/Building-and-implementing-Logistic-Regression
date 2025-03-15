You're right! I missed including the section about the creation of the logistic regression model. Below is the updated README with more detailed information on the creation of the logistic regression model:

---

# Logistic Regression from Scratch in Python

This notebook demonstrates the implementation of a **Logistic Regression** model from scratch using Python. The model is trained on the **PIMA Diabetes** dataset, and it involves implementing key components of logistic regression like the sigmoid function, cost function, gradient descent for optimization, and model evaluation.

## Overview

Logistic regression is a fundamental machine learning algorithm for binary classification tasks. It predicts the probability that a given input belongs to a particular class.

In this notebook, we:
1. Implement logistic regression from scratch using gradient descent.
2. Train the model using the PIMA Diabetes dataset.
3. Evaluate the model's performance using accuracy scores on both training and test datasets.

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

### 1. **Creating the Logistic Regression Model**

The main component of this notebook is the **Logistic_Regression** class. This class implements logistic regression from scratch using key machine learning concepts.

#### Key Components:
- **Sigmoid Function:** This function maps the model’s output (a linear combination of inputs) to a probability value between 0 and 1.

  The sigmoid function is defined as:
  \[
  \hat{Y} = \frac{1}{1 + e^{-(Xw + b)}}
  \]
  Where:
  - \( X \) is the input data.
  - \( w \) are the weights.
  - \( b \) is the bias term.
  
- **Gradient Descent:** An optimization algorithm used to minimize the cost function by updating the weights and bias. We perform the following weight updates during each iteration:

  \[
  w = w - \alpha \cdot \frac{\partial J}{\partial w}
  \]
  \[
  b = b - \alpha \cdot \frac{\partial J}{\partial b}
  \]
  Where:
  - \( \alpha \) is the learning rate.
  - \( J \) is the cost function.
  
- **Cost Function:** The cost function (also known as the loss function) used for logistic regression is the **binary cross-entropy** loss, which is optimized using gradient descent.

#### The `Logistic_Regression` Class:

```python
class Logistic_Regression:
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        # Initialize weights and bias
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        # Sigmoid function
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

        # Derivatives of the cost function
        dw = (1/self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1/self.m) * np.sum(Y_hat - self.Y)

        # Update weights and bias
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        # Predict probabilities using the sigmoid function
        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)  # Convert probabilities to binary outcomes
        return Y_pred
```

### 2. **Data Preprocessing**

The **PIMA Diabetes** dataset is loaded and preprocessed. We standardize the features (e.g., glucose levels, BMI) using `StandardScaler` to ensure that all features have the same scale, which helps improve convergence during gradient descent.

```python
# Load the dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# Separating features and labels
features = diabetes_dataset.drop(columns='Outcome', axis=1)
target = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)

# Updating features with standardized data
features = standardized_data
```

### 3. **Model Training**

After preprocessing the data, we split it into training and testing sets using **train_test_split**. We then initialize the logistic regression model and train it using the training data.

```python
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# Initialize and train the logistic regression model
classifier = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
classifier.fit(X_train, Y_train)
```

### 4. **Model Evaluation**

After training the model, we evaluate its performance by calculating the **accuracy** on both the training and testing datasets using **accuracy_score** from `sklearn.metrics`.

```python
# Accuracy on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# Accuracy on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
```

### 5. **Making Predictions**

Once the model is trained, we can use it to make predictions for new data points. The following example shows how to make predictions for a single input.

```python
# Example input data for prediction
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input data to numpy array and reshape
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_as_numpy_array)

# Make prediction
prediction = classifier.predict(std_data)

# Output result
if prediction[0] == 0:
    print("The person is not diabetic.")
else:
    print("The person is diabetic.")
```

---

## Output

### Example Output:

```plaintext
Accuracy score of the training data:  0.7735
Accuracy score of the test data:  0.758
The person is diabetic.
```

### Conclusion

This notebook demonstrates the process of implementing logistic regression from scratch using Python. It covers the steps of building the model, training it, evaluating its performance, and using it to make predictions on new data. By using this approach, you gain insight into how logistic regression works at a lower level, particularly in terms of optimization and decision boundaries.

