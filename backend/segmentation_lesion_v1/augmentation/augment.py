import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocessing.preprocess import preprocess_image, preprocess_mask, extract_roi

def apply_copy_paste(
    image: np.ndarray, 
    mask: np.ndarray, 
    lesion_bank: list[dict], 
    prob: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    if not lesion_bank or np.random.rand() > prob:
        return image, mask
        
    img_out = image.copy()
    mask_out = mask.copy()
    num_pastes = np.random.randint(1, 6)
    h, w = image.shape[:2]
    
    for _ in range(num_pastes):
        lesion_idx = np.random.randint(0, len(lesion_bank))
        lesion = lesion_bank[lesion_idx]
        lesion_img = lesion['image']
        lesion_mask = lesion['mask']
        lh, lw = lesion_img.shape[:2]
        if lh >= h or lw >= w:
            continue
        y_start = np.random.randint(0, h - lh)
        x_start = np.random.randint(0, w - lw)
        roi_img = img_out[y_start:y_start+lh, x_start:x_start+lw]
        blended = lesion_img * lesion_mask + roi_img * (1.0 - lesion_mask)
        img_out[y_start:y_start+lh, x_start:x_start+lw] = blended
        mask_out[y_start:y_start+lh, x_start:x_start+lw] = np.maximum(mask_out[y_start:y_start+lh, x_start:x_start+lw], lesion_mask)
        
    return img_out, mask_out

def create_lesion_bank(image_dir: str, mask_dir: str, target_lesion_size: int = 64) -> list[dict]:
    lesion_bank = []
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        return lesion_bank
    image_names = sorted(os.listdir(image_dir))
    mask_names = sorted(os.listdir(mask_dir))
    
    for img_name, mask_name in zip(image_names, mask_names):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img_roi, bbox = extract_roi(img)
        h_roi, w_roi = img_roi.shape[:2]
        processed_img = preprocess_image(img, target_size=(w_roi, h_roi), use_roi=True)
        processed_mask = preprocess_mask(mask, target_size=(w_roi, h_roi), bbox=bbox)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_mask.astype(np.uint8))
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w <= target_lesion_size and h <= target_lesion_size and area > 2:
                lesion_img_patch = processed_img[y:y+h, x:x+w]
                lesion_mask_patch = processed_mask[y:y+h, x:x+w]
                lesion_bank.append({'image': lesion_img_patch, 'mask': lesion_mask_patch})
                
    return lesion_bank

def get_train_augmentations(target_size: tuple[int, int]) -> A.Compose:
    return A.Compose([
        A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ToTensorV2()
    ])

def get_val_augmentations(target_size: tuple[int, int]) -> A.Compose:
    return A.Compose([
        ToTensorV2()
    ])
