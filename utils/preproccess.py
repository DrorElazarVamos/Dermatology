import cv2 as cv
import numpy as np
from PIL import Image


def convert_color_space(image, color_space: str):
    """
    Convert image to the specified color space.
    Args:
        image: BGR image (as loaded by cv2.imread)
        color_space: target color space — 'RGB', 'BGR', or 'LAB'
    Returns:
        Converted image as a numpy array.
    """
    color_space = color_space.upper().strip()

    if color_space == "BGR":
        return image.copy()
    elif color_space == "RGB":
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)
    elif color_space == "LAB":
        return cv.cvtColor(image, cv.COLOR_BGR2LAB)
    else:
        raise ValueError(f"Unsupported color space '{color_space}'. Choose from: BGR, RGB, LAB.")


def enhance_image(image):
    """
    Enhance image quality by:
      - Denoising with a bilateral filter (edge-preserving smoothing)
      - Sharpening via unsharp masking
      - CLAHE contrast enhancement on the luminance channel
    Args:
        image: BGR image as a numpy array.
    Returns:
        Enhanced BGR image.
    """
    # Edge-preserving denoising
    denoised = cv.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Unsharp masking — sharpens fine detail
    blur = cv.GaussianBlur(denoised, (0, 0), sigmaX=3)
    sharpened = cv.addWeighted(denoised, 1.5, blur, -0.5, 0)

    # CLAHE on the L channel to improve local contrast without blowing out colors
    lab = cv.cvtColor(sharpened, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv.merge([l_enhanced, a, b])

    return cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)


def create_skin_mask(image):
    """
    Create a binary mask that isolates skin and skin-infection regions.
    Combines HSV and YCrCb skin-tone ranges for robust detection, then
    applies morphological cleanup to remove noise and fill gaps.

    Args:
        image: BGR image as a numpy array.
    Returns:
        masked_image: original image with non-skin pixels set to black.
        mask: single-channel uint8 mask (255 = skin/infection, 0 = background).
    """
    # --- HSV skin range ---
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv.inRange(hsv, lower_hsv, upper_hsv)

    # --- YCrCb skin range (more robust across lighting conditions) ---
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine: a pixel must be detected as skin in both spaces
    mask = cv.bitwise_and(mask_hsv, mask_ycrcb)

    # Morphological cleanup: remove speckle noise, then fill holes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)

    # Apply mask to the original image
    masked_image = cv.bitwise_and(image, image, mask=mask)

    return masked_image, mask


def preprocess(pil_image: Image.Image) -> Image.Image:
    """
    Full preprocessing pipeline for a PIL image before CNN inference:
      1. PIL RGB → OpenCV BGR
      2. Skin/infection mask — zeros out background
      3. Image quality enhancement
      4. OpenCV BGR → PIL RGB
    Args:
        pil_image: PIL Image in RGB mode.
    Returns:
        Preprocessed PIL Image in RGB mode.
    """
    bgr = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
    masked, _ = create_skin_mask(bgr)
    enhanced = enhance_image(masked)
    return Image.fromarray(cv.cvtColor(enhanced, cv.COLOR_BGR2RGB))
