import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_pixel_metrics(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true > 0.5).astype(np.uint8)
    
    tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
    fn = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
    tn = np.sum((y_pred_bin == 0) & (y_true_bin == 0))
    
    smooth = 1e-6
    
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'sensitivity': float(recall),
        'recall': float(recall),
        'specificity': float(specificity)
    }

def compute_auprc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    try:
        from sklearn.metrics import precision_recall_curve, auc
    except ImportError:
        return 0.0

    y_true_flat = (y_true > 0.5).astype(np.uint8).flatten()
    y_pred_flat = y_pred.flatten()
    
    if np.sum(y_true_flat) == 0:
        if np.sum(y_pred_flat) == 0:
            return 1.0
        return 0.0
        
    precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_flat)
    auprc_val = auc(recall, precision)
    return float(auprc_val)

def _get_boundary(mask: np.ndarray) -> np.ndarray:
    if np.sum(mask) == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    boundary = mask.astype(bool) ^ eroded
    return boundary

def compute_surface_distances(y_pred_bin: np.ndarray, y_true_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_boundary = _get_boundary(y_pred_bin)
    true_boundary = _get_boundary(y_true_bin)
    
    if not np.any(pred_boundary) or not np.any(true_boundary):
        return np.array([]), np.array([])
        
    dist_to_true = distance_transform_edt(~true_boundary)
    distances_pred_to_gt = dist_to_true[pred_boundary]
    
    dist_to_pred = distance_transform_edt(~pred_boundary)
    distances_gt_to_pred = dist_to_pred[true_boundary]
    
    return distances_pred_to_gt, distances_gt_to_pred

def compute_boundary_metrics(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred_bin = (y_pred >= threshold).astype(bool)
    y_true_bin = (y_true > 0.5).astype(bool)
    
    d_pred_to_gt, d_gt_to_pred = compute_surface_distances(y_pred_bin, y_true_bin)
    
    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0:
        if not np.any(y_pred_bin) and not np.any(y_true_bin):
            return {'hd95': 0.0, 'assd': 0.0}
        max_dist = np.sqrt(y_pred.shape[0]**2 + y_pred.shape[1]**2)
        return {'hd95': float(max_dist), 'assd': float(max_dist)}
        
    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    hd95 = np.percentile(all_distances, 95)
    assd = (np.sum(d_pred_to_gt) + np.sum(d_gt_to_pred)) / (len(d_pred_to_gt) + len(d_gt_to_pred))
    
    return {
        'hd95': float(hd95),
        'assd': float(assd)
    }
