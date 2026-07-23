import os
import sys
import argparse

# Đảm bảo console sử dụng UTF-8
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm thư mục hiện tại vào sys.path để import cấu hình
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

from app.core.config import settings

def main():
    parser = argparse.ArgumentParser(description="Khởi tạo cơ sở dữ liệu cho dự án DR.")
    parser.add_argument(
        "--drop", 
        action="store_true", 
        help="Xóa sạch các bảng cũ của dự án trước khi khởi tạo lại (dùng khi bị xung đột cấu trúc)."
    )
    args = parser.parse_args()

    db_url = settings.DATABASE_URL
    print(f"🔌 Đang kết nối tới Supabase database: {db_url.split('@')[-1]}...")
    
    # Đường dẫn tới file init.sql
    init_sql_path = os.path.join(os.path.dirname(backend_dir), "database", "init.sql")
    if not os.path.exists(init_sql_path):
        print(f"❌ Không tìm thấy file init.sql tại: {init_sql_path}")
        return
        
    print(f"📄 Đang đọc cấu trúc SQL từ: {init_sql_path}...")
    with open(init_sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()
        
    try:
        from sqlalchemy import create_engine
        engine = create_engine(db_url)
        with engine.connect() as conn:
            dbapi_conn = conn.connection
            with dbapi_conn.cursor() as cursor:
                # Nếu có cờ --drop, thực hiện dọn dẹp các bảng cũ trước
                if args.drop:
                    print("⚠️ Đang thực hiện xóa sạch các bảng cũ của dự án (DROP TABLE ... CASCADE)...")
                    drop_queries = [
                        "DROP TABLE IF EXISTS recalls CASCADE;",
                        "DROP TABLE IF EXISTS doctor_reviews CASCADE;",
                        "DROP TABLE IF EXISTS lesion_segmentation_results CASCADE;",
                        "DROP TABLE IF EXISTS ai_results CASCADE;",
                        "DROP TABLE IF EXISTS screenings CASCADE;",
                        "DROP TABLE IF EXISTS patients CASCADE;",
                        "DROP TABLE IF EXISTS users CASCADE;"
                    ]
                    for q in drop_queries:
                        cursor.execute(q)
                    dbapi_conn.commit()
                    print("✨ Đã dọn dẹp xong các bảng cũ!")

                print("🚀 Đang khởi tạo các bảng mới và chèn dữ liệu mẫu...")
                cursor.execute(sql_content)
            dbapi_conn.commit()
        print("✅ Khởi tạo cơ sở dữ liệu Supabase thành công!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo cơ sở dữ liệu: {e}")

if __name__ == "__main__":
    main()
