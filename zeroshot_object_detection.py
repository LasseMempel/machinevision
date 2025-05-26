from transformers import pipeline
import numpy as np
from PIL import Image
from PIL import ImageDraw

checkpoint = [
    #"IDEA-Research/grounding-dino-base",
    #"google/owlv2-base-patch16-ensemble,
    #"google/owlvit-base-patch32",
    "google/owlv2-large-patch14-ensemble"
             ][0]
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")


image_path = "C:/repos/machinevision/Blockbergung Perlen_cropped.png"
image = Image.open(image_path).convert("RGB")
#image.show()

predictions = detector(
    image,
    candidate_labels=["bead"],
)

#print(predictions)

draw = ImageDraw.Draw(image)

"""
for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]
    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")
"""
"""
for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]
    xmin, ymin, xmax, ymax = box.values()

    # Compute center of the box
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Use smaller of width or height for diameter
    width = xmax - xmin
    height = ymax - ymin
    radius = min(width, height) / 2

    # Define bounding box of the circle
    ellipse_box = (cx - radius, cy - radius, cx + radius, cy + radius)

    draw.ellipse(ellipse_box, outline="red", width=2)
    draw.text((cx + radius, cy), f"{label}: {round(score, 2)}", fill="white")
"""

for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]
    xmin, ymin, xmax, ymax = box.values()

    # Draw an ellipse inside the bounding box
    ellipse_box = (xmin, ymin, xmax, ymax)
    draw.ellipse(ellipse_box, outline="red", width=2)

    # Place label just above the ellipse
    draw.text((xmin, ymin - 10), f"{label}: {round(score, 2)}", fill="white")

image.show()



