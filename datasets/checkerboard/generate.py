import enum
import os
import random
import shutil
from typing import Literal

from PIL import Image, ImageDraw


class Colors(enum.Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)

    @classmethod
    def random_color(cls):
        return random.choice(list(cls))


def generate_checkerboard(height, width, square_size):
    image = Image.new("RGB", (width * square_size, height * square_size))
    draw = ImageDraw.Draw(image)

    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                color = (255, 255, 255)  # white
            else:
                color = (0, 0, 0)  # black

            draw.rectangle(
                [
                    (j * square_size, i * square_size),
                    ((j + 1) * square_size, (i + 1) * square_size),
                ],
                fill=color,
            )

    return image


def add_shape(
    checkerboard_image,
    size_range,
    shape_type: Literal["circle", "triangle", "rectangle"],
):
    width, height = checkerboard_image.size
    draw = ImageDraw.Draw(checkerboard_image)

    # Randomly select the shape type
    # shape_type = random.choice(["oval", "circle", "triangle", "rectangle"])

    # Randomly select the size as a percentage of the original image
    size_percentage = random.uniform(*size_range)
    size = int(min(width, height) * size_percentage)

    # Randomly select the position of the shape
    x = random.randint(0, width - size)
    y = random.randint(0, height - size)

    # Randomly assign colors to the shapes
    color = Colors.random_color().value

    # Draw the shape on the checkerboard image with the randomly assigned color
    if shape_type == "circle":
        draw.ellipse([(x, y), (x + size, y + size)], fill=color)
    elif shape_type == "triangle":
        draw.polygon([(x, y), (x + size, y), (x + size // 2, y + size)], fill=color)
    elif shape_type == "rectangle":
        draw.rectangle([(x, y), (x + size, y + size)], fill=color)

    rotation = random.choice([0, 90, 180, 270])
    checkerboard_image = checkerboard_image.rotate(rotation)
    return checkerboard_image


def generate_dataset(
    N=100,
    size=(400, 400),
    output_size=(224, 224),
    square_frequency=10,
    size_range=(0.1, 0.65),
):
    dataset = []
    shape_types = ["circle", "triangle", "rectangle"]
    shapes_per_type = N // len(shape_types)

    height, width = size

    square_size = int(400 / square_frequency)
    height = width = int(height / square_size)

    for shape_type in shape_types:
        for _ in range(shapes_per_type):
            checkerboard_image = generate_checkerboard(height, width, square_size)
            image_with_shape = add_shape(checkerboard_image, size_range, shape_type)
            image_with_shape = image_with_shape.resize(output_size)
            dataset.append((image_with_shape, shape_type))

    return dataset


if __name__ == "__main__":
    save_path = "data/synthetic-checkerboard"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    N = 150
    frequency = [2, 4, 8, 16, 32, 64]

    train_data = []
    test_data = []
    for f in frequency:
        train_dataset = generate_dataset(
            N=N,
            size=(400, 400),
            output_size=(224, 224),
            square_frequency=frequency[0],
            size_range=(0.1, 0.50),
        )
        test_dataset = generate_dataset(
            N=int(0.1 * N),
            size=(400, 400),
            output_size=(224, 224),
            square_frequency=frequency[0],
            size_range=(0.20, 0.50),
        )

        train_data.extend(train_dataset)
        test_data.extend(test_dataset)

    for i, (img, shape_type) in enumerate(train_data):
        if not os.path.exists(f"{save_path}/train/{shape_type}"):
            os.makedirs(f"{save_path}/train/{shape_type}", exist_ok=True)
        img.save(f"{save_path}/train/{shape_type}/{i}.png")

    for i, (img, shape_type) in enumerate(test_data):
        if not os.path.exists(f"{save_path}/val/{shape_type}"):
            os.makedirs(f"{save_path}/val/{shape_type}", exist_ok=True)
        img.save(f"{save_path}/val/{shape_type}/{i}.png")
