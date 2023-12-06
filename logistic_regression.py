import numpy as np
from utils import generate_data
import matplotlib.pyplot as plt


def train(X_train, t_train, X_valid, t_valid):
    """
    Trains a logistic regression model using gradient descent.

    Parameters:
    - X_train: numpy array, shape (M, d), where M is the number of training samples and d is the number of features.
    - t_train: numpy array, shape (M, 1), containing the target labels for the training data.
    - X_valid: numpy array, shape (N, d), where N is the number of validation samples.
    - t_valid: numpy array, shape (N, 1), containing the target labels for the validation data.

    Returns:
    - losses: list of float, representing the cross-entropy losses for each epoch during training.
    - valid_accs: list of float, representing the validation accuracies for each epoch during training.
    - epoch_best: int, the epoch at which the highest validation accuracy is achieved.
    - acc_best: float, the highest validation accuracy achieved.
    - w_best: numpy array, shape (d, 1), the weights of the model with the highest validation accuracy.
    """

    # Initialize the weights
    d = X_train.shape[1]
    w = np.zeros([d, 1])

    # Helpers to keep track of best weights
    losses = []
    w_best = None
    valid_accs = []
    epoch_best = 0
    acc_best = 0

    # Go through each epoch
    for epoch in range(max_epoch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(X_train.shape[0] / batch_size))):
            # Get the batches
            X_batch = X_train[b * batch_size : (b + 1) * batch_size]
            t_batch = t_train[b * batch_size : (b + 1) * batch_size]

            # Find the loss for this batch
            loss_batch, _ = predict(X_batch, w, t_batch)
            loss_this_epoch += loss_batch

            # Update the weights
            X_batch_t = np.transpose(X_batch)
            z_batch = np.dot(X_batch, w)
            y = 1 / (1 + np.exp(-z_batch))
            gradient = (1 / X_batch.shape[0]) * (
                np.dot(X_batch_t, (y - t_batch))
            ) + 2 * decay * w
            w = w - alpha * gradient

        # Monitor model performance

        # Sum up the Cross-Entropy Loss
        loss = loss_this_epoch
        losses.append(loss)

        # Find validation accuracy
        _, acc = predict(X_valid, w, t_valid)
        valid_accs.append(acc)

        # Keep track of best epoch
        if acc > acc_best:
            epoch_best = epoch
            acc_best = acc
            w_best = w

    return losses, valid_accs, epoch_best, acc_best, w_best


def predict(X, w, t):
    """
    Predicts labels using a logistic regression model and calculates the cross-entropy loss and accuracy.

    Parameters:
    - X: numpy array, shape (M, d), where M is the number of samples and d is the number of features.
    - w: numpy array, shape (d, 1), the weights of the logistic regression model.
    - t: numpy array, shape (M, 1), containing the true labels.

    Returns:
    - loss: float, the cross-entropy loss for the predictions on the given data.
    - acc: float, the accuracy of the predictions on the given data.
    """

    y = 1 / (1 + np.exp(-1 * np.dot(X, w)))
    y = (y >= 0.5).astype(int)  # predict y is 1 if >= 0.5

    # Obtain Cross Entropy Loss
    epsilon = 1e-15
    loss = -np.mean(t * np.log(y + epsilon) + (1 - t) * np.log(1 - y + epsilon))

    # Obtain Accuracy
    equals = np.sum(np.equal(t, y))
    acc = equals / t.shape[0]

    return loss, acc


# Main Program

max_epoch = 1000
batch_size = 10
decay = 0.00
alpha = 0.001


X_train, X_valid, X_test, t_train, t_valid, t_test = generate_data()
losses, valid_accs, epoch_best, acc_best, w_best = train(
    X_train, t_train, X_valid, t_valid
)
# print(epoch_best)

_, test_acc = predict(X_test, w_best, t_test)
print("Test Accuracy : ", test_acc)

# plt.figure()
# plt.scatter(range(max_epoch), losses)
# plt.xlabel("Epochs")
# plt.ylabel("Training Loss")
# plt.savefig("Training_losses.jpg")

# plt.figure()
# plt.scatter(range(max_epoch), valid_accs)
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.savefig("Validation_accuracies.jpg")
