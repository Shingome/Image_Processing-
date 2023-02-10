import numpy as np
from PIL import Image


# size
width = 600
height = 600

# open image file
image = Image.open("img.jpg")
image = image.crop((0, 0, width, height))
image = image.convert('L')
image.show()

image_array = []

# image_array to arrays
for x in range(width):
    for y in range(height):
        crop = image.crop((x, y, x + 8, y + 8))
        image_array.append(np.reshape(np.asarray(crop) / 255, (1, 64)))

# save image_array
image_array = np.asarray(image_array)

np.save('../training/image', image_array)

print(np.shape(image_array))
