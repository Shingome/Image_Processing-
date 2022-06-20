def create_map(image_array):
    import numpy as np

    # Load synapses
    synapses = np.load('synapses.npz')
    W1 = synapses['arr_0']
    b1 = synapses['arr_1']
    W2 = synapses['arr_2']
    b2 = synapses['arr_3']
    W3 = synapses['arr_4']
    b3 = synapses['arr_5']

    def predict(x):
        def relu(t):
            return np.maximum(t, 0)

        def softmax(t):
            out = np.exp(t)
            return out / np.sum(out)

        # Calculate
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        h2 = relu(t2)
        t3 = h2 @ W3 + b3
        z = softmax(t3)
        return z

    # Form map
    map = []
    for x in image_array:
        z = predict(x)
        y_pred = np.argmax(z)
        map.append(y_pred)
    return map
