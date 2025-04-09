from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os

# Directory to save generated digit images
output_dir = "font_based_digits"
os.makedirs(output_dir, exist_ok=True)

# Path to font file (adjust this to a valid .ttf file path on your system)
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"  # Replace with the path to your desired font
font_size = 29  # Adjust font size to control the size of the digits

# Adjustable vertical offset (positive moves text down, negative moves it up)
vertical_offset = 7  # Try values like 2, 3, or higher to move text downward

# Create images for digits 0 to 9
digits = range(10)
digit_images = []

for digit in digits:
    # Create a blank 28x28 grayscale image (MNIST size)
    img = Image.new("L", (28, 28), color=0)  # Black background
    draw = ImageDraw.Draw(img)

    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Could not load font at {font_path}. Using default font.")
        font = ImageFont.load_default()

    # Get text size to center the digit
    text_bbox = font.getbbox(str(digit))  # Get bounding box (left, top, right, bottom)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = (28 - text_width) // 2  # Center horizontally
    text_y = (28 - text_height) // 2  # Center vertically

    # Adjust text_y for proper vertical alignment and apply vertical offset
    ascent, descent = font.getmetrics()  # Get font ascent and descent
    text_y -= (text_bbox[1] + ascent - descent) // 2
    text_y += vertical_offset  # Move text down by adding the offset

    # Draw the digit on the image
    draw.text((text_x, text_y), str(digit), fill=255, font=font)  # White digit
    digit_images.append(np.array(img))  # Store the image for visualization

    # Save the image
    img.save(os.path.join(output_dir, f"digit_{digit}.png"))

# Display the generated images
plt.figure(figsize=(10, 5))
for i, digit_img in enumerate(digit_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digit_img, cmap="gray")
    plt.axis("off")
    plt.title(f"Digit {i}")

plt.tight_layout()
# plt.show()

output_dir = "scip/data/ri/mnist"  # Directory for saving the images
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

for digit, avg_img in enumerate(digit_images):
    # Save each image as a PNG file
    image_path = os.path.join(output_dir, f"{digit}.png")
    Image.fromarray(avg_img).save(image_path)

