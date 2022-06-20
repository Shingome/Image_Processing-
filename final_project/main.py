from create_map import create_map
from draw_image import draw_image
from prepare_image import prepare_image
import gradio as gr


def image_mod(image):
    return draw_image(create_map(prepare_image(image)), image.size)


iface = gr.Interface(image_mod, gr.inputs.Image(type="pil"), "image")

iface.launch(share=True)
