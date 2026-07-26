import cv2
import numpy as np

def extract_roi(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = image.shape[:2]
        return image, (0, h, 0, w)
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = image[y:y+h, x:x+w]
    
    max_side = max(w, h)
    padded = np.zeros((max_side, max_side, 3), dtype=image.dtype)
    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    return padded, (y, y+h, x, x+w)

def apply_ben_graham(image: np.ndarray, sigma: int = 10) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    normalized = cv2.addWeighted(image, 4, blur, -4, 128)
    return normalized

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced

def preprocess_image(image: np.ndarray, target_size: tuple[int, int] = (512, 512), use_roi: bool = True) -> np.ndarray:
    if use_roi:
        img_roi, _ = extract_roi(image)
    else:
        img_roi = image.copy()
        
    green_ch = img_roi[:, :, 1]
    graham_img = apply_ben_graham(green_ch, sigma=10)
    enhanced_img = apply_clahe(graham_img, clip_limit=2.0, tile_grid_size=(8, 8))
    
    resized_img = cv2.resize(enhanced_img, target_size, interpolation=cv2.INTER_CUBIC)
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    return normalized_img

def preprocess_mask(mask: np.ndarray, target_size: tuple[int, int] = (512, 512), bbox: tuple[int, int, int, int] = None) -> np.ndarray:
    if bbox is not None:
        y_min, y_max, x_min, x_max = bbox
        h, w = y_max - y_min, x_max - x_min
        cropped = mask[y_min:y_max, x_min:x_max]
        
        max_side = max(w, h)
        padded = np.zeros((max_side, max_side), dtype=mask.dtype)
        y_offset = (max_side - h) // 2
        x_offset = (max_side - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
        mask_roi = padded
    else:
        mask_roi = mask.copy()
        
    resized_mask = cv2.resize(mask_roi, target_size, interpolation=cv2.INTER_NEAREST)
    
    max_val = mask_roi.max()
    if max_val > 0:
        binary_mask = (resized_mask >= (max_val / 2.0)).astype(np.float32)
    else:
        binary_mask = np.zeros_like(resized_mask, dtype=np.float32)
    
    return binary_mask
