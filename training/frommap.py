from PIL import Image, ImageDraw
import numpy as np


step = 5
width = 600
height = 600

width *= step
height *= step

image = Image.new('RGB', (width, height), (255, 255, 255))
draw = ImageDraw.Draw(image)

map = np.load('map.npy')

print(len(map))

iter = 0

k = 8

for x in range(0, width, step):
    for y in range(0, height, step):
        if map[iter] == 1:
            xn, yn = x, y + k
        elif map[iter] == 2:
            xn, yn = x + k, y
        elif map[iter] == 3:
            xn, yn = x + k, y - k
        elif map[iter] == 4:
            xn, yn = x + k, y + k
        else:
            iter += 1
            continue
        draw.line(xy=[(x, y), (xn, yn)], fill='black')
        iter += 1

image.show()
