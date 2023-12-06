import numpy as np
import math


def generate_data():
    """
    Generates and prepares training, validation, and test data from CSV files.

    Returns:
    - X_train: numpy array, shape (train_M, d), the feature matrix for training data.
    - X_valid: numpy array, shape (valid_M, d), the feature matrix for validation data.
    - X_test: numpy array, shape (test_M, d), the feature matrix for test data.
    - t_train: numpy array, shape (train_M, 1), the target matrix for training data.
    - t_valid: numpy array, shape (valid_M, 1), the target matrix for validation data.
    - t_test: numpy array, shape (test_M, 1), the target matrix for test data.
    """

    # feature_file = "endgamedata.csv"
    feature_file = "midgamedata.csv"
    target_file = "results.csv"

    # Read from csv files and create feature and target matrices
    X = np.genfromtxt(feature_file, delimiter=",", skip_header=1)
    t = np.genfromtxt(target_file, delimiter=",", skip_header=1).astype(int)

    # Add extra 1 to absorb bias
    M = X.shape[0]
    X = np.concatenate([X, np.ones([M, 1])], axis=1)
    t = np.reshape(t, (t.shape[0], 1))

    # Shuffle the data
    np.random.seed(314)
    np.random.shuffle(X)
    np.random.seed(314)
    np.random.shuffle(t)

    # Upper limits for training and validation
    train_M = math.ceil(0.6 * M)
    valid_M = train_M + math.ceil(0.2 * M)

    X_train = X[:train_M]
    X_valid = X[train_M:valid_M]
    X_test = X[valid_M:]

    t_train = t[:train_M]
    t_valid = t[train_M:valid_M]
    t_test = t[valid_M:]

    return X_train, X_valid, X_test, t_train, t_valid, t_test
