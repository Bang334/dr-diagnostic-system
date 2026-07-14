import os
import subprocess
import argparse
import zipfile
import sys

# Đảm bảo in ra console không bị lỗi Unicode trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def create_directory_structure(base_dir):
    """Tạo cấu trúc thư mục chuẩn cho dữ liệu DR Grading."""
    print(f"[*] Đang tạo cấu trúc thư mục tại: {base_dir}")
    dirs_to_create = [
        "raw/aptos2019",
        "raw/eyepacs",
        "processed/aptos2019/train_images",
        "processed/aptos2019/test_images",
        "processed/eyepacs/train_images",
        "processed/eyepacs/test_images",
        "splits" # Lưu trữ các file csv đã split train/val/test
    ]
    
    for d in dirs_to_create:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"  + {path}")
    print("[v] Đã hoàn tất tạo cấu trúc thư mục.\n")

def check_kaggle_auth():
    """Kiểm tra cấu hình Kaggle API."""
    home_dir = os.path.expanduser('~')
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("[!] CẢNH BÁO: Không tìm thấy file ~/.kaggle/kaggle.json")
        print("Để sử dụng tính năng tải tự động, vui lòng thực hiện:")
        print("  1. Đăng nhập vào kaggle.com -> Settings -> Create New Token")
        print("  2. Đặt file kaggle.json tải về vào thư mục: " + kaggle_dir)
        print("  3. Chạy lại script này với cờ --aptos hoặc --eyepacs.\n")
        return False
    return True

def download_dataset(competition_name, raw_dir, display_name):
    """Tải và giải nén dữ liệu từ Kaggle."""
    print(f"[*] Đang tải {display_name} từ Kaggle...")
    try:
        # Cần cài đặt kaggle package: pip install kaggle
        subprocess.run([
            "kaggle", "competitions", "download", 
            "-c", competition_name, 
            "-p", raw_dir
        ], check=True)
        
        # Tìm file zip để giải nén
        zip_path = os.path.join(raw_dir, f"{competition_name}.zip")
        if os.path.exists(zip_path):
            print(f"[*] Đang giải nén {display_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            os.remove(zip_path) # Xóa file zip để tiết kiệm dung lượng
            
        print(f"[v] Hoàn tất chuẩn bị {display_name}!\n")
    except subprocess.CalledProcessError:
        print(f"[x] Lỗi: Kaggle từ chối truy cập. Bạn đã vào trang {competition_name} và bấm 'I Understand and Accept' rules chưa?\n")
    except FileNotFoundError:
        print("[x] Lỗi: Không tìm thấy lệnh 'kaggle'. Vui lòng chạy: pip install kaggle\n")
    except Exception as e:
        print(f"[x] Lỗi không xác định khi tải {display_name}: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script tải và cấu trúc dữ liệu cho bài toán DR Grading")
    parser.add_argument("--aptos", action="store_true", help="Tự động tải dữ liệu APTOS 2019 (~8.2GB)")
    parser.add_argument("--eyepacs", action="store_true", help="Tự động tải dữ liệu EyePACS (~82GB)")
    args = parser.parse_args()

    # Xác định thư mục data (đặt ở root dự án)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "data")
    
    print("=== SETUP DỮ LIỆU DR GRADING (THÀNH VIÊN 1) ===\n")
    
    # 1. Luôn tạo cấu trúc thư mục
    create_directory_structure(data_dir)
    
    # 2. Xử lý tải dữ liệu
    if args.aptos or args.eyepacs:
        if check_kaggle_auth():
            if args.aptos:
                download_dataset("aptos2019-blindness-detection", os.path.join(data_dir, "raw", "aptos2019"), "APTOS 2019")
            if args.eyepacs:
                print("[!] LƯU Ý: Dữ liệu EyePACS rất lớn (~82GB). Quá trình tải có thể mất nhiều giờ.")
                download_dataset("diabetic-retinopathy-detection", os.path.join(data_dir, "raw", "eyepacs"), "EyePACS")
    else:
        print("[*] LƯU Ý: Không có tham số tải tự động (--aptos hoặc --eyepacs).")
        print("Để tải dữ liệu tự động, hãy chạy:")
        print("  python download_data.py --aptos")
        print("  python download_data.py --eyepacs")
        print("Hoặc bạn có thể tự tải và giải nén thủ công vào các thư mục 'raw' tương ứng.")
