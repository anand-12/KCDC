import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def estimate_conditional_mean(X, Y, kernel):
    """
    Estimate the conditional mean function of X given Y or Y given X
    using the training data.
    """
    kernel_XY = pairwise_kernels(X, Y, metric=kernel)
    kernel_YY = pairwise_kernels(Y, Y, metric=kernel)
    alpha = np.linalg.solve(kernel_YY, kernel_XY.T)
    if X.shape[1] > 1:
        return alpha.T
    else:
        return alpha.ravel()


def calculate_rkhs_mean_embedding(X_test, X_train, Y_train, kernel):
    """
    Calculate the RKHS mean embedding of X|Y and Y|X using the test data.
    """
    kernel_XX = pairwise_kernels(X_test, X_train, metric=kernel)
    cond_mean = estimate_conditional_mean(X_train, Y_train, kernel)
    cond_mean = cond_mean.reshape(500, 500)
    return kernel_XX.dot(cond_mean)


# Example usage
sigma_sq = 1
X = np.random.normal(loc = 0, size=(1000, 1))
Y = 2 * X + np.random.normal(loc = 0, scale=sigma_sq, size=(1000, 1))

# Split the data into training and test sets
num_samples = 500
X_train, X_test = X[:num_samples], X[num_samples:]
Y_train, Y_test = Y[:num_samples], Y[num_samples:]

# Calculate the RKHS mean embedding of X|Y and Y|X using a Gaussian kernel
kernel = 'rbf'
X_given_Y_embedding = calculate_rkhs_mean_embedding(X_test, X_train, Y_train, kernel)
Y_given_X_embedding = calculate_rkhs_mean_embedding(Y_test, Y_train, X_train, kernel)

print('RKHS mean embedding of X|Y:', X_given_Y_embedding)
print('RKHS mean embedding of Y|X:', Y_given_X_embedding)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate synthetic data
X_given_Y_embedding = calculate_rkhs_mean_embedding(X_test, X_train, Y_train, kernel)
Y_given_X_embedding = calculate_rkhs_mean_embedding(Y_test, Y_train, X_train, kernel)
X_Y_embeddings = np.concatenate((X_given_Y_embedding, Y_given_X_embedding), axis=0)
labels = np.concatenate((np.ones(num_samples), np.zeros(num_samples)), axis=0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_Y_embeddings, labels, test_size=0.3)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the classifier on the test data
score = clf.score(X_test, y_test)
print("Accuracy:", score)
