# Building and Implementing Logistic Regression from Scratch in Python

This repository contains an implementation of **Linear Regression** from scratch in Python using **Gradient Descent** for model training. This approach will help you understand the internal workings of Linear Regression without relying on pre-built machine learning libraries. The main objective of this project is to predict the salary of individuals based on their years of experience using linear regression.

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Data Processing](#data-processing)
4. [Model Implementation](#model-implementation)
5. [Training the Model](#training-the-model)
6. [Making Predictions](#making-predictions)
7. [Visualization](#visualization)
8. [Key Concepts](#key-concepts)

## Introduction

This project implements **Linear Regression** with **Gradient Descent** for training. Linear Regression tries to model the relationship between a dependent variable (e.g., salary) and one or more independent variables (e.g., years of experience). In this specific case, we predict the salary based on the work experience.

The formula for **Linear Regression** is:

```
Y = wX + b
```

Where:
- **Y** is the dependent variable (target, e.g., salary).
- **X** is the independent variable (feature, e.g., years of experience).
- **w** is the weight (parameter learned during training).
- **b** is the bias (intercept).

We use **Gradient Descent** to minimize the loss function and update the model's parameters, **w** and **b**.

## Dependencies

To run the code, you will need the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`

These can be installed using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Data Processing

The data used in this example is stored in a CSV file called `salary_data.csv`. The dataset contains the following columns:

- **Years of Experience** (independent variable `X`).
- **Salary** (dependent variable `Y`).

Steps for data processing:

1. **Loading the Data**: The data is read from a CSV file using `pandas.read_csv()`.
2. **Inspecting the Data**: The first few rows are displayed using `.head()`, and the shape of the dataset is checked using `.shape()`.
3. **Missing Values**: Missing values are checked with `.isnull().sum()`.

```python
salary_data = pd.read_csv('/content/salary_data.csv')
salary_data.head()
```

## Model Implementation

### Linear Regression Class

The model is implemented in the `Linear_Regression` class. It contains the following methods:

1. **`__init__(learning_rate, no_of_iterations)`**: Initializes the learning rate and the number of iterations for training.
2. **`fit(X, Y)`**: Trains the model using gradient descent to optimize weights `w` and bias `b`.
3. **`update_weights()`**: Updates the parameters `w` and `b` by computing gradients and applying the gradient descent rule.
4. **`predict(X)`**: Makes predictions using the learned parameters.

### Gradient Descent Formula

- **Update Rule for Weights (`w`)**:
  ```
  w = w - α * dw
  ```
- **Update Rule for Bias (`b`)**:
  ```
  b = b - α * db
  ```

Where:
- **α** is the learning rate.
- **dw** and **db** are the gradients of the loss function with respect to **w** and **b**.

## Training the Model

To train the model:

1. Split the dataset into training and test sets using `train_test_split()` from `sklearn`.
2. Create an instance of the `Linear_Regression` class and call the `fit()` method to train the model on the training data.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=2)

model = Linear_Regression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X_train, Y_train)
```

## Making Predictions

Once the model is trained, we can make predictions on the test data by calling the `predict()` method:

```python
test_data_prediction = model.predict(X_test)
```

The predicted salary values will be compared to the actual values to evaluate the performance of the model.

## Visualization

The predicted and actual values are visualized using **Matplotlib**. A scatter plot of the test data (actual values) is drawn, and the regression line (predicted values) is plotted over it.

```python
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, test_data_prediction, color='blue')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()
```

This visualization helps in understanding how well the model fits the data.

## Key Concepts

### Gradient Descent

- **Gradient Descent** is an optimization algorithm used to minimize the cost function (mean squared error in this case).
- The model parameters (weights and bias) are updated iteratively in the direction opposite to the gradient of the cost function with respect to the parameters.
  
### Learning Rate

- **Learning rate (α)** is a hyperparameter that determines the step size during each update of the parameters.
- A small learning rate leads to slow convergence, while a large learning rate can cause overshooting of the minimum.

### Loss Function

The loss function used in this implementation is the **Mean Squared Error (MSE)**, which is given by:

```
MSE = (1/m) * Σ (Y_i - Y_pred)^2
```

Where:
- **m** is the number of data points.
- **Y_i** is the true value of the target variable.
- **Y_pred** is the predicted value.

---

By building linear regression from scratch, you gain insight into how machine learning models work under the hood. This process also helps in better understanding optimization algorithms like gradient descent, which are central to many machine learning techniques.

#   B u i l d i n g - a n d - i m p l e m e n t i n g - L o g i s t i c - R e g r e s s i o n  
 