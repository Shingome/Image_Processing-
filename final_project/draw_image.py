def draw_image(map, size):
    from PIL import Image, ImageDraw

    # size
    step = 10
    width, height = size
    width = width // 8 * 8 * step
    height = height // 8 * 8 * step

    # create canvas
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    iter = 0

    # drawing
    for x in range(0, width, step):
        for y in range(0, height, step):
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

    return image
