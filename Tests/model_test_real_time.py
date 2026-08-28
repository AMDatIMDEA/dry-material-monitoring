import cv2
import platform
import time
from pathlib import Path
from ultralytics import YOLO


# ================================
# User settings
# ================================

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "Material_level_using_yolo"
    / "model_weights"
    / "polymer_empty.pt"
)

CAMERA_INDEX = 1
# Usually:
# 0 = laptop/internal camera
# 1 = first external camera
# 2 = second external camera

CONFIDENCE = 0.50
# Start confidence threshold.
# Example:
# 0.25 = detects more objects, but more false detections
# 0.70 = stricter, fewer detections

IMG_SIZE = 640

DEVICE = "cpu"
# Use "cpu" if you do not have NVIDIA GPU.
# Use 0 if you have CUDA NVIDIA GPU.


# ================================
# Check model path
# ================================

model_path = Path(MODEL_PATH).resolve()

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")


# ================================
# Load YOLO model
# ================================

print("Loading model...")
model = YOLO(str(model_path))
print("Model loaded successfully.")

print("Class names:")
print(model.names)


# ================================
# Open camera
# ================================

cap = (
    cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if platform.system() == "Windows"
    else cv2.VideoCapture(CAMERA_INDEX)
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera index {CAMERA_INDEX}. "
        "Try CAMERA_INDEX = 0, 1, or 2."
    )

# Optional camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera opened.")
print("Controls:")
print("  q  = quit")
print("  +  = increase confidence")
print("  -  = decrease confidence")


# ================================
# FPS variables
# ================================

prev_time = time.perf_counter()
fps_smooth = 0.0
confidence = CONFIDENCE


# ================================
# Main loop
# ================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera.")
        break

    start_time = time.perf_counter()

    # YOLO inference
    results = model.predict(
        source=frame,
        conf=confidence,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    result = results[1]

    # Draw segmentation masks, boxes, and labels
    annotated_frame = result.plot()

    # Calculate FPS
    end_time = time.perf_counter()
    current_fps = 1.0 / max(end_time - start_time, 1e-6)

    # Smooth FPS so it does not jump too much
    fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

    # Count detections
    num_detections = 0
    if result.boxes is not None:
        num_detections = len(result.boxes)

    # Display information
    cv2.putText(
        annotated_frame,
        f"FPS: {fps_smooth:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Confidence: {confidence:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Detections: {num_detections}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    # Show frame
    cv2.imshow("YOLO Segmentation Camera Test", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("+") or key == ord("="):
        confidence = min(confidence + 0.05, 0.95)
        print(f"Confidence increased to {confidence:.2f}")

    elif key == ord("-") or key == ord("_"):
        confidence = max(confidence - 0.05, 0.05)
        print(f"Confidence decreased to {confidence:.2f}")


# ================================
# Cleanup
# ================================

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")
