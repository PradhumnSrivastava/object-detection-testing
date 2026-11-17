
# ==========================================
# SIMPLE YOLO OBJECT DETECTION PROJECT
# ==========================================

# Import required libraries
from ultralytics import YOLO
from pathlib import Path

# ------------------------------------------
# STEP 1: Define file paths
# ------------------------------------------

image_path = Path("input/input.jpg")
output_path = Path("output/detected_image.jpg")

# ------------------------------------------
# STEP 2: Check input image
# ------------------------------------------

if not image_path.exists():
    raise FileNotFoundError("Input image not found!")

# ------------------------------------------
# STEP 3: Create output folder
# ------------------------------------------

output_path.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------
# STEP 4: Load YOLOv8 Nano model
# ------------------------------------------

model = YOLO("yolov8n.pt")

# ------------------------------------------
# STEP 5: Perform object detection
# ------------------------------------------

results = model.predict(
    source=str(image_path),
    imgsz=320,
    conf=0.35,
    device="cpu",
    max_det=20,
    verbose=False
)

# ------------------------------------------
# STEP 6: Get detection results
# ------------------------------------------

result = results[0]

# ------------------------------------------
# STEP 7: Save detected image
# ------------------------------------------

result.save(filename=str(output_path))

# ------------------------------------------
# STEP 8: Display detected objects
# ------------------------------------------

print("\nDETECTED OBJECTS")
print("-------------------------")

for box in result.boxes:

    # Get class ID
    class_id = int(box.cls[0])

    # Get confidence score
    confidence = float(box.conf[0])

    # Get object name
    object_name = model.names[class_id]

    print(f"Object: {object_name}")
    print(f"Confidence: {confidence:.2f}")
    print("-------------------------")

# ------------------------------------------
# STEP 9: Completion message
# ------------------------------------------

print("\nObject Detection Completed!")
print("Output Image:", output_path)
