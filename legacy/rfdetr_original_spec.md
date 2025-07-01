# RF-DETR Predict Method - Complete Implementation Guide

This document provides a comprehensive analysis of the `predict()` method in the RF-DETR library, including all dependencies, helper functions, and implementation details needed to recreate the functionality.

## Overview

The `predict()` method is defined in the `RFDETR` class at `/usr/local/lib/python3.12/dist-packages/rfdetr/detr.py:114-166`. It performs object detection inference on a single image and returns standardized detection results.

## Method Signature

```python
def predict(
    self,
    image_or_path: Union[str, Image.Image, np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    **kwargs,
):
```

### Parameters
- **image_or_path**: Input image in various formats (file path, PIL Image, numpy array, or torch tensor)
- **threshold**: Confidence threshold for filtering detections (default: 0.5)
- **kwargs**: Additional arguments (currently unused)

### Returns
- **sv.Detections**: Supervision library detection object with bounding boxes, class IDs, and confidence scores

## Complete Implementation Breakdown

### 1. Model Setup and Inference Mode

```python
self.model.model.eval()  # Set model to evaluation mode
with torch.inference_mode():  # Disable gradient computation for inference
```

**Key Components:**
- `self.model`: Instance of `Model` class from `rfdetr.main`
- `self.model.model`: The actual LWDETR neural network (`LWDETR` class)
- `torch.inference_mode()`: More efficient than `torch.no_grad()` for inference

### 2. Image Loading and Preprocessing

#### 2.1 Handle Different Input Types

```python
# Handle string path input
if isinstance(image_or_path, str):
    image_or_path = Image.open(image_or_path)  # Load as PIL Image
    w, h = image_or_path.size  # Get original dimensions (PIL uses width, height)

# Convert to tensor if not already
if not isinstance(image_or_path, torch.Tensor):
    image = F.to_tensor(image_or_path)  # Convert to tensor (C, H, W), [0,1]
    _, h, w = image.shape  # Get dimensions from tensor (height, width)
else:
    # Handle pre-existing tensor input  
    # BUG: logger.warning has formatting issue
    logger.warning(
        "image_or_path is a torch.Tensor\n",
        "we expect an image divided by 255 at (C, H, W)",
    )
    assert image_or_path.shape[0] == 3, "image must have 3 channels"
    h, w = image_or_path.shape[1:]  # Extract height, width
    # BUG: Missing line: image = image_or_path
```

**CRITICAL BUG DETECTED:** The source code is missing `image = image_or_path` in the tensor input branch (line 135 in detr.py). This causes `UnboundLocalError: cannot access local variable 'image'` when tensor input is used. PIL Image and numpy array inputs work correctly.

#### 2.2 Image Preprocessing Pipeline

```python
# Move to model device (GPU/CPU)
image = image.to(self.model.device)

# Normalize using ImageNet statistics
image = F.normalize(image, self.means, self.stds)

# Resize to model resolution
image = F.resize(image, (self.model.resolution, self.model.resolution))
```

**Normalization Constants:**
```python
means = [0.485, 0.456, 0.406]  # ImageNet RGB channel means
stds = [0.229, 0.224, 0.225]   # ImageNet RGB channel standard deviations
```

### 3. Model Inference

```python
# Add batch dimension and run inference
predictions = self.model.model.forward(image[None, :])
```

**Model Forward Pass:**
- Input: `(1, 3, resolution, resolution)` tensor
- Output: Dictionary with keys:
  - `'pred_logits'`: Class predictions `(1, num_queries, num_classes+1)` - includes background class
  - `'pred_boxes'`: Bounding box predictions `(1, num_queries, 4)` in cxcywh format
  - `'aux_outputs'`: Auxiliary outputs from intermediate decoder layers
  - `'enc_outputs'`: Encoder outputs (for two-stage models)

**Important Note:** The model outputs 300 queries by default (`num_queries=300`), and the number of classes includes the background class (+1).

### 4. Post-Processing

```python
# Extract bounding boxes
bboxes = predictions["pred_boxes"]

# Apply post-processing to convert to final format
results = self.model.postprocessors["bbox"](
    predictions,
    target_sizes=torch.tensor([[h, w]], device=self.model.device),
)
```

**Post-Processor Details:**
- Converts from cxcywh to xyxy format
- Scales to original image dimensions
- Applies top-k selection (default: 100 detections from 300 queries)
- Returns list of dictionaries with 'scores', 'labels', 'boxes'

**Key Implementation Details:**
- The postprocessor selects top-100 detections from 300 model queries
- Scores are computed using sigmoid activation on logits
- Box coordinates are converted from normalized [0,1] to absolute pixel coordinates
- Labels are class indices (0-based, excluding background class)

### 5. Result Extraction and Filtering

```python
# Extract results from post-processor output
scores, labels, boxes = [], [], []
for result in results:
    scores.append(result["scores"])
    labels.append(result["labels"])  
    boxes.append(result["boxes"])

# Stack into tensors
scores = torch.stack(scores)  # (1, num_detections)
labels = torch.stack(labels)  # (1, num_detections)
boxes = torch.stack(boxes)    # (1, num_detections, 4)

# Apply confidence threshold filtering
keep_inds = scores > threshold
boxes = boxes[keep_inds]      # (n_filtered, 4)
labels = labels[keep_inds]    # (n_filtered,)
scores = scores[keep_inds]    # (n_filtered,)
```

### 6. Create Supervision Detection Object

```python
detections = sv.Detections(
    xyxy=boxes.cpu().numpy(),      # Convert to numpy array
    class_id=labels.cpu().numpy(), # Convert to numpy array
    confidence=scores.cpu().numpy(), # Convert to numpy array
)
return detections
```

## Dependencies and Required Classes

### 1. Model Class (`rfdetr.main.Model`)

```python
class Model:
    def __init__(self, **kwargs):
        # Initialize model components
        self.resolution = args.resolution  # Image resolution (e.g., 560)
        self.model = build_model(args)     # LWDETR instance
        self.device = torch.device(args.device)  # Device placement
        self.postprocessors = build_criterion_and_postprocessors(args)[1]
        
    # Key attributes used in predict():
    # - self.model: LWDETR neural network
    # - self.device: torch.device for tensor placement
    # - self.resolution: int, target image size
    # - self.postprocessors: dict with 'bbox' key
```

### 2. LWDETR Neural Network (`rfdetr.models.lwdetr.LWDETR`)

```python
class LWDETR(nn.Module):
    def forward(self, samples):
        # Process input through backbone and transformer
        # Returns dictionary with:
        return {
            'pred_logits': outputs_class[-1],  # (B, num_queries, num_classes)
            'pred_boxes': outputs_coord[-1]    # (B, num_queries, 4) in cxcywh
        }
```

### 3. PostProcess Class (`rfdetr.models.postprocessor.PostProcess`)

```python
class PostProcess(nn.Module):
    def forward(self, outputs, target_sizes):
        # Convert model outputs to detection format
        # Apply top-k selection and coordinate scaling
        # Returns list of dicts: [{'scores': ..., 'labels': ..., 'boxes': ...}]
```

### 4. Supervision Detections (`supervision.Detections`)

```python
class Detections:
    def __init__(self, xyxy, class_id=None, confidence=None):
        # xyxy: (n, 4) numpy array - bounding boxes in [x1,y1,x2,y2] format
        # class_id: (n,) numpy array - class indices
        # confidence: (n,) numpy array - confidence scores [0,1]
```

## Key Transform Functions

### 1. `torchvision.transforms.functional.to_tensor()`
- Converts PIL Image/numpy array to torch tensor
- Scales values from [0,255] to [0,1]
- Changes shape from (H,W,C) to (C,H,W)

### 2. `torchvision.transforms.functional.normalize()`
- Applies per-channel normalization: `(pixel - mean) / std`
- Uses ImageNet statistics for RGB channels
- Formula: `normalized = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`

### 3. `torchvision.transforms.functional.resize()`
- Resizes image to target dimensions
- Maintains aspect ratio or forces exact size
- Default interpolation: bilinear

## Complete Workflow Diagram

```
Input Image (str/PIL/numpy/tensor)
    ↓
[Load if string path] → PIL Image
    ↓
[to_tensor] → Tensor (C,H,W) [0,1]
    ↓
[to device] → Tensor on GPU/CPU
    ↓
[normalize] → Normalized tensor (ImageNet stats)
    ↓
[resize] → Tensor (C, resolution, resolution)
    ↓
[add batch dim] → Tensor (1, C, resolution, resolution)
    ↓
[model forward] → {'pred_logits': (1,Q,C), 'pred_boxes': (1,Q,4)}
    ↓
[postprocess] → [{'scores': (N,), 'labels': (N,), 'boxes': (N,4)}]
    ↓
[threshold filter] → Filtered tensors
    ↓
[to numpy] → numpy arrays
    ↓
[sv.Detections] → Supervision detection object
```

## Usage Example

```python
from rfdetr.detr import RFDETRBase
import supervision as sv

# Initialize model
model = RFDETRBase(pretrain_weights="path/to/weights.pth")

# Run inference
detections = model.predict("image.jpg", threshold=0.5)

# Access results
print(f"Found {len(detections)} objects")
print(f"Bounding boxes: {detections.xyxy}")      # (n, 4) array
print(f"Class IDs: {detections.class_id}")       # (n,) array  
print(f"Confidence: {detections.confidence}")    # (n,) array

# Visualize results
annotator = sv.BoxAnnotator()
annotated_image = annotator.annotate(image, detections)
```

## Error Handling and Edge Cases

1. **Invalid image paths**: `Image.open()` will raise FileNotFoundError
2. **Tensor input validation**: Asserts 3-channel input for tensor inputs
3. **Device mismatch**: Tensors automatically moved to model device
4. **Empty detections**: Returns empty sv.Detections object if no objects above threshold
5. **Memory management**: Uses `torch.inference_mode()` for efficient inference

## Performance Considerations

- **Batch processing**: Method processes single images; for multiple images, call in loop
- **Memory efficiency**: Uses inference mode to reduce memory overhead
- **GPU acceleration**: Automatically uses model's device (GPU if available)
- **Image preprocessing**: Resize operation may introduce artifacts for very different aspect ratios

## Actual Test Results

The documentation has been verified through actual testing with a trained RF-DETR model:

```
Model loaded successfully with weights from: /workspace/outputs/simple_1/checkpoint_best_total.pth
Model device: cuda, Resolution: 560
Test successful with detection results:
- Number of detections: 1 object above threshold 0.5
- Detection format: supervision.Detections object
- Bounding boxes: (1, 4) numpy array
- Class IDs: (1,) numpy array  
- Confidence scores: (1,) numpy array
- Sample detection: Class 6, Confidence 0.898
```

**Preprocessing verification:**
- PIL image (420, 560) → tensor (3, 560, 420) 
- Normalization range: [-2.118, 2.429] (ImageNet normalized)
- Resize to: (3, 560, 560) square format
- Model forward: 300 queries → top-100 postprocessed detections

## Critical Source Code Issues Discovered

During detailed analysis, two bugs were discovered in the original RF-DETR source code:

### 1. **Missing Variable Assignment (Line 135 in detr.py)**
```python
# Current broken code:
else:
    logger.warning(...)
    assert image_or_path.shape[0] == 3, "image must have 3 channels"
    h, w = image_or_path.shape[1:]
    # Missing: image = image_or_path

# Fixed version should be:
else:
    logger.warning(...)
    assert image_or_path.shape[0] == 3, "image must have 3 channels"
    h, w = image_or_path.shape[1:]
    image = image_or_path  # This line is missing!
```

**Impact**: Tensor input causes `UnboundLocalError` on line 137 when `image.to(device)` is called.

### 2. **Logger Formatting Issue (Line 130-133 in detr.py)**
```python
# Current problematic code:
logger.warning(
    "image_or_path is a torch.Tensor\n",
    "we expect an image divided by 255 at (C, H, W)",
)

# Should be:
logger.warning(
    "image_or_path is a torch.Tensor\n"
    "we expect an image divided by 255 at (C, H, W)"
)
```

**Impact**: Causes `TypeError: not all arguments converted during string formatting` in logging.

## Workaround
Currently, the predict method only works with:
- String paths (✅)
- PIL Images (✅) 
- Numpy arrays (✅)
- Torch tensors (❌ - causes error)

This documentation provides all the information needed to recreate the RF-DETR predict functionality from scratch, including all dependencies, data flows, implementation details, and known bugs.