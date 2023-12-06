import numpy as np
from utils import generate_data
import matplotlib.pyplot as plt


def train(X_train, t_train, X_valid, t_valid, n_classes):
    """
    Trains a softmax regression model using gradient descent.

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
    - w_best: numpy array, shape (d, n_classes), the weights of the model with the highest validation accuracy.
    """

    # W : d x k
    # z : XW : M x k
    # y : M x k

    # Initialize the weights
    W = np.zeros([X_train.shape[1], n_classes])

    # Helpers to keep track of the best weights
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
            loss_batch, _ = predict(X_batch, W, t_batch)
            loss_this_epoch += loss_batch

            # Get the one hot encoded target matrix
            t_encoded = one_hot_encode(t_batch, n_classes)

            # Get the softmax(XW)
            y_batch = softmax(X_batch, W)

            # Update the weights
            gradient = (1 / X_batch.shape[0]) * (
                np.dot(np.transpose(X_batch), (y_batch - t_encoded))
            ) + 2 * decay * W
            W = W - alpha * gradient

        # Monitor behaviour after each epoch

        # Add losses to loss history
        losses.append(loss_this_epoch)

        # Calculate validation accuracy
        _, acc = predict(X_valid, W, t_valid)
        valid_accs.append(acc)

        # Keep track of best epoch
        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            W_best = W

    return losses, valid_accs, epoch_best, acc_best, W_best


def softmax(X, W):
    """
    Calculates the softmax activation for a given input.

    Parameters:
    - X: numpy array, shape (M, d), where M is the number of samples and d is the number of features.
    - W: numpy array, shape (d, k), where k is the number of classes.

    Returns:
    - y: numpy array, shape (M, k), the softmax activations for each class.
    """

    # Obtain the shifted softmax input
    z = np.dot(X, W)
    z_max = np.amax(z, axis=1, keepdims=True)
    z = z - z_max

    # Get softmax z
    y = np.exp(z)
    y_sum = np.sum(y, axis=1, keepdims=1)
    y = y / y_sum

    return y


def one_hot_encode(t, n_classes):
    """
    One-hot encodes the target labels.

    Parameters:
    - t: numpy array, shape (M, 1), containing the target labels.
    - n_classes: int, the number of classes.

    Returns:
    - t_encoded: numpy array, shape (M, k), the one-hot encoded target matrix.
    """

    t = np.reshape(t, (t.shape[0],))

    # One hot encode the y labels
    t_encoded = np.zeros([t.shape[0], n_classes])
    t_encoded[np.arange(t.shape[0]), t] = 1

    return t_encoded


def predict(X, W, t):
    """
    Predicts labels using a softmax regression model and calculates the cross-entropy loss and accuracy.

    Parameters:
    - X: numpy array, shape (M, d), where M is the number of samples and d is the number of features.
    - w: numpy array, shape (d, 1), the weights of the softmax regression model.
    - t: numpy array, shape (M, 1), containing the true labels.

    Returns:
    - loss: float, the cross-entropy loss for the predictions on the given data.
    - acc: float, the accuracy of the predictions on the given data.
    """

    # Get softmax(XW)
    y = softmax(X, W)
    y_hat = np.argmax(y, axis=1)

    # Calculate accuracy
    t_reshaped = np.reshape(t, (t.shape[0],))
    m = y_hat.shape[0]
    acc = np.sum(y_hat == t_reshaped) / m

    # Calculate loss
    loss = -np.mean(np.log(y[np.arange(len(t_reshaped)), t_reshaped]))

    return loss, acc


# Main Program

max_epoch = 1000
batch_size = 50
decay = 0.0
alpha = 0.001
n_classes = 2

X_train, X_valid, X_test, t_train, t_valid, t_test = generate_data()
losses, valid_accs, epoch_best, acc_best, w_best = train(
    X_train, t_train, X_valid, t_valid, n_classes
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
