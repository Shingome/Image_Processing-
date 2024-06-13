import numpy as np
import matplotlib.pyplot as plt


# Hyper arguments
INPUT_DIM = 64
OUT_DIM = 5
H1_DIM = 16
H2_DIM = 10

ALPHA = 0.005
NUM_EPOCHS = 100

# Global
img_map = []
loss_arr = []


def relu(t):
    return np.maximum(t, 0)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def relu_deriv(t):
    return (t >= 0).astype(float)


def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    h2 = relu(t2)
    t3 = h2 @ W3 + b3
    z = softmax(t3)
    return z


def calc_accuracy():
    correct = 0
    for y, x in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc


def create_map():
    for x in image:
        z = predict(x)
        y_pred = np.argmax(z)
        img_map.append(y_pred)


if __name__ == "__main__":
    # Load infomation
    dataset = np.load('../dataset.npy', allow_pickle=True)

    image = np.load('./../image.npy')

    # Random synapses
    W1 = np.random.rand(INPUT_DIM, H1_DIM)
    b1 = np.random.rand(1, H1_DIM)
    W2 = np.random.rand(H1_DIM, H2_DIM)
    b2 = np.random.rand(1, H2_DIM)
    W3 = np.random.rand(H2_DIM, OUT_DIM)
    b3 = np.random.rand(1, OUT_DIM)

    W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
    b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
    W2 = (W2 - 0.5) * 2 * np.sqrt(1/H1_DIM)
    b2 = (b2 - 0.5) * 2 * np.sqrt(1/H1_DIM)
    W3 = (W3 - 0.5) * 2 * np.sqrt(1/H2_DIM)
    b3 = (b3 - 0.5) * 2 * np.sqrt(1/H2_DIM)

    loss = 0

    # Backpropagation
    for ep in range(NUM_EPOCHS):
        print(ep)
        np.random.shuffle(dataset)
        for i in range(len(dataset)):

            x = dataset[i][1]
            y = dataset[i][0]

            # Forward
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            h2 = relu(t2)
            t3 = h2 @ W3 + b3
            z = softmax(t3)
            E = sparse_cross_entropy(z, y)

            # Backward
            y_full = to_full(y, OUT_DIM)
            dE_dt3 = z - y_full
            dE_dW3 = h2.T @ dE_dt3
            dE_db3 = np.sum(dE_dt3, axis=0, keepdims=True)
            dE_dh2 = dE_dt3 @ W3.T
            dE_dt2 = dE_dh2 * relu_deriv(t2)
            dE_dW2 = h1.T @ dE_dt2
            dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
            dE_dh1 = dE_dt2 @ W2.T
            dE_dt1 = dE_dh1 * relu_deriv(t1)
            dE_dW1 = x.T @ dE_dt1
            dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

            # Update
            W1 = W1 - ALPHA * dE_dW1
            b1 = b1 - ALPHA * dE_db1
            W2 = W2 - ALPHA * dE_dW2
            b2 = b2 - ALPHA * dE_db2
            W3 = W3 - ALPHA * dE_dW3
            b3 = b3 - ALPHA * dE_db3

            loss += E

        loss_arr.append(loss)
        loss = 0

    # Accuracy
    accuracy = calc_accuracy()
    print("Accuracy:", accuracy)

    # Map
    create_map()
    np.save('map', np.asarray(img_map))

    # Plot
    plt.plot(loss_arr)
    plt.show()

    # Save synapses
    np.savez('./../synapses', W1, b1, W2, b2, W3, b3)
