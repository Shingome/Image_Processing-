import os.path
import PIL
import tensorflow_hub as hub

from src.neural_style_transfer import StyleStealer

from src.predict import *

import src.predict_old as old


def numpy_predict(image):
    return old.draw_image(old.create_map(old.prepare_image(image)), image.size)


def keras_predict(image, batch_size, step, line_size, model_id):
    model_filename = os.path.join("models", models[model_id], "weights.keras")
    return process_image(image, batch_size, step, line_size, model_filename)


def nst_predict(content_img, style_variant):
    style_path = os.path.join("images", style_images[style_variant])
    return worker.steal(content_img, np.array(PIL.Image.open(style_path)))


def video_predict(video, batch_size, step, line_size, model_id):
    model_filename = os.path.join("models", models[model_id], "weights.keras")
    frames = slice_video(video)
    output_file = create_video_from_frames(video, frames, batch_size, step, line_size, model_filename)
    return output_file


with gr.Blocks() as demo:
    batches = [2 ** i for i in range(0, 18)]

    models = {
        "1": "adam_1_64_000000",
        "10": "adam_10_64_000001",
        "50": "adam_50_64_000003",
        "100": "adam_100_64_000004",
        "500": "adam_500_64_000005",
        "1000": "adam_1000_64_000006",
        "5000": "adam_5000_64_000007",
        "10000": "adam_10000_64_000008"
    }

    models_choices = list(models.keys())

    style_images = {
        "v1": "style_image_1.jpg",
        "v2": "style_image_2.jpg",
        "v3": "style_image_3.jpg",
        "v4": "style_image_4.jpeg",
        "v5": "style_image_5.jpeg",
        "v6": "style_image_6.jpg"
    }

    style_variants = list(style_images.keys())

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    worker = StyleStealer(hub_model)

    with gr.Tab("Numpy"):
        with gr.Row():
            n_image_input = gr.Image(sources=["upload", "webcam"], type="pil")
            n_image_output = gr.Image(sources=None)
        n_submit_button = gr.Button("Submit")
    with gr.Tab("Keras"):
        with gr.Row():
            with gr.Column():
                k_image_input = gr.Image(sources=["upload", "webcam"], type="pil")
                k_batch_size = gr.Dropdown(
                    batches[:-2], label="Batch_size", value=64, info="Влияет на скорость обработки изображения"
                )
                k_step = gr.Slider(
                    minimum=1, maximum=8, label="Step", value=8, step=1, info="Расстояние между блоками"
                )
                k_line_size = gr.Slider(
                    minimum=1, maximum=16, label="Line_size", value=8, step=1, info="Длина штриха"
                )
                k_weights = gr.Radio(
                    models_choices,
                    value=models_choices[3],
                    label="Weights",
                    info="Количество эпох обучения"
                )
            k_image_output = gr.Image()
        k_submit_button = gr.Button("Submit")
    with gr.Tab("NST"):
        with gr.Row():
            with gr.Column():
                nst_image_input = gr.Image(sources=["upload", "webcam"], type="pil")
                nst_style_variant = gr.Radio(
                    style_variants,
                    value=style_variants[0],
                    label="Variants"
                )
            nst_image_output = gr.Image()
        nst_submit_button = gr.Button("Submit")
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                v_image_input = gr.Video(sources=["upload"])
                v_batch_size = gr.Dropdown(
                    batches, label="Batch_size", value=batches[15], info="Влияет на скорость обработки изображения"
                )
                v_step = gr.Slider(
                    minimum=1, maximum=8, label="Step", value=8, step=1, info="Расстояние между блоками"
                )
                v_line_size = gr.Slider(
                    minimum=1, maximum=16, label="Line_size", value=8, step=1, info="Длина штриха"
                )
                v_weights = gr.Radio(
                    models_choices,
                    value=models_choices[3],
                    label="Weights",
                    info="Количество эпох обучения"
                )
            v_image_output = gr.Video()
        v_submit_button = gr.Button("Submit")

        n_submit_button.click(numpy_predict, inputs=n_image_input, outputs=n_image_output)

        k_submit_button.click(keras_predict,
                              inputs=[k_image_input, k_batch_size, k_step, k_line_size, k_weights],
                              outputs=k_image_output)

        nst_submit_button.click(
            nst_predict,
            inputs=[nst_image_input, nst_style_variant],
            outputs=nst_image_output)

        v_submit_button.click(video_predict,
                              inputs=[v_image_input, v_batch_size, v_step, v_line_size, v_weights],
                              outputs=v_image_output)

demo.launch()
