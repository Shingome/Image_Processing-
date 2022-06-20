import numpy as np
from PIL import Image


# open image_for_dataset file
image_for_dataset = Image.open("img_for_dataset.jpg")

# change image
width, height = image_for_dataset.size
image_for_dataset = image_for_dataset.crop((0, 0, width // 8 * 8, height // 8 * 8))
image_for_dataset = image_for_dataset.convert('L')
image_for_dataset.show()

images = []

# images to arrays
for x in range(width):
    for y in range(height):
        image = image_for_dataset.crop((x, y, x + 8, y + 8))
        images.append(np.asarray(image))

# save crops_for_dataset
images_for_dateset = np.asarray(images)

print(np.shape(images))

np.save('images_for_dataset', images)
