from src.predict import *
import gradio as gr


def predict(image):
    return draw_image(create_map(prepare_image(image)), image.size)


if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=["image"]
    )

    iface.launch(share=True)
