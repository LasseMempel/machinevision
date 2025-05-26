from transformers import pipeline
from PIL import Image, ImageDraw

# Load model
checkpoint = "google/owlv2-large-patch14-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

# Load target image (containing many beads)
main_image = Image.open("C:/repos/machinevision/pictures/Blockbergung Perlen_cropped.png").convert("RGB")

# Load example images (queries) — one image per bead color
query_images = [
    Image.open("queries/red_bead_cropped.png").convert("RGB"),
    #Image.open("queries/white_bead_cropped.png").convert("RGB"),
    Image.open("queries/green_bead_cropped.png").convert("RGB")
]

query_labels = [
    "red bead", 
    #"white bead", 
    "green bead"  # Optional, for labeling the result
    ]

# Run detector with image queries
predictions = detector(
    main_image,
    image_queries=query_images,
    candidate_labels=query_labels,  # Optional but recommended for human-readable labels
)

# Draw results
draw = ImageDraw.Draw(main_image)

for pred in predictions:
    box = pred["box"]
    label = pred["label"]
    score = pred["score"]
    xmin, ymin, xmax, ymax = box.values()

    draw.ellipse((xmin, ymin, xmax, ymax), outline="red", width=2)
    draw.text((xmin, ymin - 10), f"{label}: {round(score, 2)}", fill="white")

main_image.show()