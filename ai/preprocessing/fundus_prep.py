import cv2
import numpy as np
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse

def crop_image_from_gray(img, tol=7):
    """
    Cắt bỏ phần viền đen thừa xung quanh ảnh đáy mắt.
    Dựa trên việc tìm mask của các pixel có giá trị > tol.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img # Toàn ảnh đen, trả về ảnh gốc
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.dstack([img1, img2, img3])
        return img

def preprocess_fundus_image(image_path, img_size=512):
    """
    Thực hiện pipeline tiền xử lý:
    1. Cắt viền đen
    2. Resize
    3. Trích xuất kênh Green
    4. CLAHE
    5. Ben Graham
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # 1. Cắt viền đen và Resize
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (img_size, img_size))
    
    # 2. Trích xuất kênh Green
    # OpenCV đọc ảnh theo thứ tự BGR
    b, g, r = cv2.split(img)
    
    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh Green
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g)
    
    # 4. Ben Graham Preprocessing
    # Công thức: original * 4 - gaussian_blur * 4 + 128
    # Kỹ thuật này giúp khử sáng không đều và làm nổi bật mạch máu/tổn thương
    gaussian = cv2.GaussianBlur(g_clahe, (0, 0), sigmaX=30)
    img_bg = cv2.addWeighted(g_clahe, 4, gaussian, -4, 128)
    
    # 5. Nhân bản lên 3 kênh để tương thích với input của các mạng CNN (ResNet, EfficientNet,...)
    final_img = cv2.merge([img_bg, img_bg, img_bg])
    
    return final_img

def process_single_file(args):
    src_path, dst_path, img_size = args
    # Nếu file đã tồn tại thì bỏ qua (hỗ trợ resume khi bị đứt gánh)
    if os.path.exists(dst_path):
        return True
        
    img = preprocess_fundus_image(src_path, img_size)
    if img is not None:
        cv2.imwrite(dst_path, img)
        return True
    return False

def batch_process(src_dir, dst_dir, img_size=512, ext='*.png', num_workers=4):
    """
    Xử lý song song toàn bộ ảnh trong thư mục.
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    # Hỗ trợ nhiều định dạng ảnh (.png, .jpeg, .jpg)
    search_path = os.path.join(src_dir, ext)
    image_paths = glob.glob(search_path)
    
    if len(image_paths) == 0:
        print(f"[!] Không tìm thấy ảnh nào tại: {search_path}")
        return
        
    print(f"[*] Tìm thấy {len(image_paths)} ảnh. Đang tiến hành tiền xử lý ({img_size}x{img_size})...")
    
    tasks = []
    for src_path in image_paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, filename)
        tasks.append((src_path, dst_path, img_size))
        
    # Xử lý đa luồng giúp CPU chạy 100% thay vì chạy 1 luồng rất lâu
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Processing"):
            if result:
                success_count += 1
                
    print(f"[v] Hoàn tất! Đã xử lý thành công {success_count}/{len(tasks)} ảnh.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script tiền xử lý hàng loạt ảnh đáy mắt (DR Grading).")
    parser.add_argument("--src", type=str, required=True, help="Thư mục chứa ảnh gốc (raw)")
    parser.add_argument("--dst", type=str, required=True, help="Thư mục lưu ảnh đã xử lý (processed)")
    parser.add_argument("--size", type=int, default=512, help="Kích thước ảnh đầu ra (mặc định: 512)")
    parser.add_argument("--ext", type=str, default="*.png", help="Định dạng ảnh cần tìm (VD: *.png, *.jpeg)")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng CPU sử dụng (mặc định: 4)")
    
    args = parser.parse_args()
    batch_process(args.src, args.dst, args.size, args.ext, args.workers)
