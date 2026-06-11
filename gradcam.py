"""
gradcam.py  —  FYP2 Grad-CAM Explainable AI Module
====================================================
Generates heatmap visualizations showing which facial regions
the model focuses on when predicting each emotion.

Grad-CAM (Gradient-weighted Class Activation Mapping):
  - Computes gradients of the predicted class score
    with respect to the last convolutional layer's feature maps
  - Weights each feature map by its importance to the prediction
  - Projects the weighted activation back onto the input image
  - Result: heatmap showing which face regions influenced the prediction

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations
           from Deep Networks via Gradient-based Localization"

Usage:
    from gradcam import generate_gradcam
    heatmap_img = generate_gradcam(model, img_array, model_type, class_idx)
"""

import numpy as np
import cv2
import tensorflow as tf


# Last convolutional layer names for each model
# These are the layers whose activations we visualize
LAST_CONV_LAYERS = {
    "MobileNetV2": "Conv_1",           # Last conv layer in MobileNetV2
    "DenseNet121": "conv5_block16_concat",  # Last conv layer in DenseNet121
}


def get_gradcam_heatmap(model, img_array, model_type, class_idx):
    """
    Compute Grad-CAM heatmap for a given image and class.

    Args:
        model      : Loaded Keras model
        img_array  : Preprocessed input array shape (1, H, W, 3)
        model_type : "MobileNetV2" or "DenseNet121"
        class_idx  : Index of predicted class to explain

    Returns:
        heatmap: numpy array (H, W) normalized 0-1
    """
    last_conv_layer = LAST_CONV_LAYERS.get(model_type)
    if last_conv_layer is None:
        return None

    try:
        # Build a model that outputs both the last conv layer and predictions
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [
                model.get_layer(last_conv_layer).output,
                model.output
            ]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            inputs        = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss          = predictions[:, class_idx]

        # Gradient of the class score w.r.t. conv layer output
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)

        # Normalize to 0-1
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def apply_gradcam_overlay(original_img_bgr, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img_bgr: Original BGR image (H, W, 3)
        heatmap         : Grad-CAM heatmap (h, w) normalized 0-1
        alpha           : Heatmap opacity (0=invisible, 1=full)

    Returns:
        overlay_rgb: RGB image with heatmap overlay
    """
    if heatmap is None:
        return cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)

    h, w = original_img_bgr.shape[:2]

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to colormap (jet: blue=low, red=high attention)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend with original image
    overlay = cv2.addWeighted(original_img_bgr, 1 - alpha,
                               heatmap_colored, alpha, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def generate_gradcam(model, face_img_bgr, model_type, class_idx):
    """
    Full Grad-CAM pipeline: preprocess → compute → overlay.

    Args:
        model        : Loaded Keras model
        face_img_bgr : Cropped face region in BGR format
        model_type   : "MobileNetV2" or "DenseNet121"
        class_idx    : Index of predicted class

    Returns:
        overlay_rgb  : RGB image with Grad-CAM heatmap overlay
        heatmap      : Raw heatmap array (for debugging)
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess
    from tensorflow.keras.applications.densenet import preprocess_input as dn_preprocess

    if model_type == "MobileNetV2":
        size       = (160, 160)
        preprocess = mv2_preprocess
    else:
        size       = (224, 224)
        preprocess = dn_preprocess

    # Preprocess for model input
    resized     = cv2.resize(face_img_bgr, size)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x           = preprocess(resized_rgb.astype('float32'))
    x           = np.expand_dims(x, axis=0)

    # Generate heatmap
    heatmap = get_gradcam_heatmap(model, x, model_type, class_idx)

    # Apply overlay on the resized face (not full frame)
    overlay = apply_gradcam_overlay(resized, heatmap, alpha=0.45)

    return overlay, heatmap