import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from perceptron import Perceptron

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Parameters:
    y_true (ndarray): True class labels.
    y_pred (ndarray): Predicted class labels.

    Returns:
    float: Accuracy score.
    """
    return np.sum(y_true == y_pred) / len(y_true)

# Generate synthetic data for binary classification
X, y = make_blobs(n_samples=170, n_features=2, centers=2, cluster_std=1.05, random_state=2)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Initialize and train the Perceptron model
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
# Make predictions on the test set
predictions = p.predict(X_test)

# Calculate performance metrics using static methods from the Perceptron class
prec = Perceptron.precision(y_test, predictions)
rec = Perceptron.recall(y_test, predictions)
f1 = Perceptron.f1_score(y_test, predictions)

# Output performance metrics
print("Perceptron classification accuracy:", accuracy(y_test, predictions))
print("Perceptron classification precision:", prec)
print("Perceptron classification recall:", rec)
print("Perceptron classification F1-score:", f1)

# Plot the training data points
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, label='Training data')

# Calculate and plot the decision boundary
x0_min, x0_max = np.amin(X_train[:, 0]), np.amax(X_train[:, 0])
x1_min = (-p.weight[0] * x0_min - p.bias) / p.weight[1]
x1_max = (-p.weight[0] * x0_max - p.bias) / p.weight[1]
plt.plot([x0_min, x0_max], [x1_min, x1_max], 'b--', label='Decision boundary')

# Set y-axis limits and labels
y_min, y_max = np.amin(X_train[:, 1]), np.amax(X_train[:, 1])
plt.ylim([y_min - 3, y_max + 3])
plt.title('Perceptron Classifier Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Display the plot
plt.show()
