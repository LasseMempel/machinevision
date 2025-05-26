from transformers import pipeline
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

# Load model
checkpoint = "google/owlv2-large-patch14-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

# Load image
image_path = "C:/repos/machinevision/pictures/Blockbergung Perlen_cropped.png"
image = Image.open(image_path).convert("RGB")

# Run detection
candidate_labels = ["red bead", "blue bead", "green bead", "yellow bead", "orange bead", "purple bead", "white bead", "black bead"]
predictions = detector(image, candidate_labels=candidate_labels)

# Group predictions by bounding box (rounded to avoid float imprecision)
box_groups = defaultdict(list)

for pred in predictions:
    # Round box coordinates to 1 decimal to group similar boxes
    box_key = tuple(round(v, 1) for v in pred["box"].values())
    box_groups[box_key].append(pred)

# Keep only the highest-scoring label per box
best_predictions = []
for box_key, preds in box_groups.items():
    best_pred = max(preds, key=lambda x: x["score"])
    best_predictions.append(best_pred)

# Draw results
draw = ImageDraw.Draw(image)
for pred in best_predictions:
    box = pred["box"]
    label = pred["label"]
    score = pred["score"]
    xmin, ymin, xmax, ymax = box.values()

    draw.ellipse((xmin, ymin, xmax, ymax), outline="red", width=2)
    draw.text((xmin, ymin - 10), f"{label}: {round(score, 2)}", fill="white")

# Show image
image.show()
