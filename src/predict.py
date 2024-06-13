import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras import models
import os
import time
import cv2
import gradio as gr
import tempfile


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.6f} seconds to execute.")
        return result

    return wrapper


@timer
def prepare_image(image):
    if type(image) is np.ndarray:
        image = Image.fromarray(image)

    image = image.convert('L')
    width, height = image.size

    image_array = np.asarray(image).T

    # image to arrays
    stride_array = np.lib.stride_tricks.sliding_window_view(image_array, (8, 8))

    map_size = stride_array.shape[:2]

    # save image_array
    stride_array = stride_array.reshape(-1, 64)

    return stride_array, map_size, (width, height)


@timer
def create_map(model_path, image_array, batch_size):
    model = models.load_model(model_path)
    image = np.reshape(np.asarray(image_array), (-1, 8, 8, 1))
    return np.argmax(model.predict(image, batch_size=batch_size), axis=1)


@timer
def draw_image(image_map, map_size, img_size, step=8, line_size=8):
    # size
    width, height = map_size
    width *= step
    height *= step

    image = Image.new('RGB', (img_size[0] * step, img_size[1] * step), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    iteration = 0

    # drawing
    for x in range(0, width, step):
        for y in range(0, height, step):
            if image_map[iteration] == 1:
                xn, yn = x, y + line_size
            elif image_map[iteration] == 2:
                xn, yn = x + line_size, y
            elif image_map[iteration] == 3:
                xn, yn = x + line_size, y - line_size
            elif image_map[iteration] == 4:
                xn, yn = x + line_size, y + line_size
            else:
                iteration += 1
                continue
            draw.line(xy=[(x, y), (xn, yn)], fill='black')
            iteration += 1

    return image


@timer
def process_image(image, batch_size, step, line_size, model_filename, new_size=None):
    image_arr, map_size, img_size = prepare_image(image)
    image_map = create_map(model_filename, image_arr, batch_size)
    new_image = draw_image(image_map, map_size, img_size, step, line_size)
    if new_size is not None:
        return cv2.resize(np.asarray(new_image), new_size, cv2.INTER_LINEAR)
    return new_image


@timer
def slice_video(video_path):
    video_capture = cv2.VideoCapture(video_path)

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = np.empty((frame_count, int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.uint8)

    frame_index = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[frame_index] = gray_frame
        frame_index += 1

    video_capture.release()

    frames = np.array(frames)

    return frames

@timer
def create_video_from_frames(orig_video, frames, *args):
    progress = gr.Progress()
    cap = cv2.VideoCapture(orig_video)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    size = (int(width), int(height))

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, size)

    for frame in progress.tqdm(frames, desc="Processing"):
        frame = np.array(process_image(frame, *args, size))
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()

    result_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    os.system(f"ffmpeg -i {video_path} -vcodec libx264 {result_path}")

    return result_path

