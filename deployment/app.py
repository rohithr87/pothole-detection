'import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
def create_model():
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model

print("Loading model...")
model = create_model()
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# Preprocessing transform
transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def detect_potholes(image, confidence_threshold):
    """Detect potholes in the uploaded image"""
    if image is None:
        return None, "Please upload an image"

    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Store original dimensions
    original_image = image.copy()
    original_h, original_w = image.shape[:2]

    # Preprocess image
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)[0]

    # Filter predictions by confidence threshold
    keep = predictions['scores'] >= confidence_threshold
    boxes = predictions['boxes'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()

    # Scale boxes back to original image size
    scale_x = original_w / 640
    scale_y = original_h / 640

    if len(boxes) > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    # Draw bounding boxes on the image
    result_image = original_image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        # Draw rectangle
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # Draw label
        label = f"{score:.0%}"
        cv2.putText(result_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Create summary text
    num_detections = len(boxes)
    if num_detections > 0:
        summary = f"Found {num_detections} pothole(s):\\n\\n"
        for i, score in enumerate(scores, 1):
            summary += f"  Pothole {i}: {score:.1%} confidence\\n"
    else:
        summary = "No potholes detected in this image.\\n\\nTry:\\n- Lowering the confidence threshold\\n- Using a clearer image"

    return result_image, summary

# Create Gradio interface (Gradio 3.x syntax)
demo = gr.Interface(
    fn=detect_potholes,
    inputs=[
        gr.Image(type="numpy", label="Upload Road Image"),
        gr.Slider(
            minimum=0.1,
            maximum=0.9,
            value=0.5,
            step=0.05,
            label="Confidence Threshold"
        )
    ],
    outputs=[
        gr.Image(type="numpy", label="Detection Result"),
        gr.Textbox(label="Detection Summary", lines=5)
    ],
    title="🚗 Pothole Detection System",
    description="""
## AI-Powered Road Damage Detection

Upload an image of a road to automatically detect potholes using deep learning.

**Model:** Faster R-CNN with ResNet50-FPN backbone | **AP@50:** 62.9% | **Recall:** 76.5%
    """,
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)