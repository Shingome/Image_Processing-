import numpy as np
from PIL import Image, ImageDraw
import gradio as gr


def prepare_image(image: Image):
    progress = gr.Progress()

    # convert image
    width, height = image.size
    width = width // 8 * 8
    height = height // 8 * 8
    image = image.crop((0, 0, width, height))
    image = image.convert('L')

    image_array = []

    # image to arrays
    for x in progress.tqdm(range(width), desc="Preparing"):
        for y in range(height):
            crop = image.crop((x, y, x + 8, y + 8))
            image_array.append(np.reshape(np.asarray(crop) / 255, (1, 64)))

    # save image_array
    image_array = np.asarray(image_array)

    return image_array


def draw_image(map, size):
    progress = gr.Progress()

    # size
    step = 10
    width, height = size
    new_width = width // 8 * 8 * step
    new_height = height // 8 * 8 * step

    # create canvas
    image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    iter = 0

    # drawing
    for x in progress.tqdm(range(0, new_width, step), desc="Drawing"):
        for y in range(0, new_height, step):
            if map[iter] == 1:
                xn, yn = x, y + 8
            elif map[iter] == 2:
                xn, yn = x + 8, y
            elif map[iter] == 3:
                xn, yn = x + 8, y - 8
            elif map[iter] == 4:
                xn, yn = x + 8, y + 8
            else:
                iter += 1
                continue
            draw.line(xy=[(x, y), (xn, yn)], fill='black')
            iter += 1

    image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image


def create_map(image_array):
    # Load synapses
    synapses = np.load('./final_synapses.npz')
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

    progress = gr.Progress()

    # Form map
    map = []
    for x in progress.tqdm(image_array, desc="Processing"):
        z = predict(x)
        y_pred = np.argmax(z)
        map.append(y_pred)
    return map