def prepare_image(image):
    import numpy as np

    # open image file
    width, height = image.size
    width = width // 8 * 8
    height = height // 8 * 8
    image = image.crop((0, 0, width, height))
    image = image.convert('L')

    image_array = []

    # image to arrays
    for x in range(width):
        for y in range(height):
            crop = image.crop((x, y, x + 8, y + 8))
            image_array.append(np.reshape(np.asarray(crop) / 255, (1, 64)))

    # save image_array
    image_array = np.asarray(image_array)

    return image_array
