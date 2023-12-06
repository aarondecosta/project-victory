import numpy as np
from utils import generate_data
import matplotlib.pyplot as plt


def train(X_train, t_train, X_valid, t_valid):
    """
    Trains a linear regression model using gradient descent.

    Parameters:
    - X_train: numpy array, shape (M, d), where M is the number of training samples and d is the number of features.
    - t_train: numpy array, shape (M, 1), containing the target labels for the training data.
    - X_valid: numpy array, shape (N, d), where N is the number of validation samples.
    - t_valid: numpy array, shape (N, 1), containing the target labels for the validation data.

    Returns:
    - losses: list of float, representing the MSE losses for each epoch during training.
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
        # Keep track of loss for each epoch
        loss_this_epoch = 0

        # Update weights for each batch
        for b in range(int(np.ceil(X_train.shape[0] / batch_size))):
            # Get batches
            X_batch = X_train[b * batch_size : (b + 1) * batch_size]
            t_batch = t_train[b * batch_size : (b + 1) * batch_size]

            # Find loss for this batch
            loss_batch, _ = predict(X_batch, w, t_batch)
            loss_this_epoch += loss_batch

            # Update weights
            X_batch_t = np.transpose(X_batch)
            gradient = (1 / X_batch.shape[0]) * (
                np.dot(np.dot(X_batch_t, X_batch), w) - np.dot(X_batch_t, t_batch)
            ) + 2 * decay * w
            w = w - alpha * gradient

        # Monitor model performance

        # Average out the squared loss
        loss = loss_this_epoch / (b + 1)
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
    Predicts labels using a linear regression model and calculates the MSE loss and accuracy.

    Parameters:
    - X: numpy array, shape (M, d), where M is the number of samples and d is the number of features.
    - w: numpy array, shape (d, 1), the weights of the linear regression model.
    - t: numpy array, shape (M, 1), containing the true labels.

    Returns:
    - loss: float, the cross-entropy loss for the predictions on the given data.
    - acc: float, the accuracy of the predictions on the given data.
    """

    y = np.dot(X, w)
    y = (y >= 0.5).astype(int)  # predict y is 1 if >= 0.5

    # Obtain Mean Squared Loss
    diff = y - t
    loss = (1 / (2 * y.shape[0])) * (np.dot(np.transpose(diff), diff)).item(0, 0)

    # Obtain Accuracy
    equals = np.sum(np.equal(t, y))
    acc = equals / t.shape[0]

    return loss, acc


# Main Program

max_epoch = 1000
batch_size = 10
decay = 0.01
alpha = 0.0001


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
