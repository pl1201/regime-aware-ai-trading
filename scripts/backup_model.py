"""
Script backup model hàng ngày
"""
import os
import shutil
import datetime
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backup_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Đường dẫn đến model
MODEL_PATH = "models/dynamic_moe_v2_enhanced_final.pkl"
BACKUP_DIR = "models/backups"

# Tạo thư mục backup nếu chưa có
os.makedirs(BACKUP_DIR, exist_ok=True)

# Kiểm tra model có tồn tại không
if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ Model không tồn tại tại {MODEL_PATH}")
    exit(1)

# Tạo tên file backup với timestamp
now = datetime.datetime.now()
backup_filename = f"dynamic_moe_v2_enhanced_final_{now.strftime('%Y%m%d')}.pkl"
backup_path = os.path.join(BACKUP_DIR, backup_filename)

# Sao chép model
try:
    shutil.copy2(MODEL_PATH, backup_path)
    logger.info(f"✅ Đã backup model thành công: {backup_path}")
except Exception as e:
    logger.error(f"❌ Lỗi khi backup model: {e}")
    exit(1)

# Xóa các file backup cũ hơn 30 ngày
thirty_days_ago = now - datetime.timedelta(days=30)
for filename in os.listdir(BACKUP_DIR):
    if filename.startswith("dynamic_moe_v2_enhanced_final_") and filename.endswith(".pkl"):
        file_path = os.path.join(BACKUP_DIR, filename)
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        if file_time < thirty_days_ago:
            try:
                os.remove(file_path)
                logger.info(f"🗑️  Đã xóa backup cũ: {file_path}")
            except Exception as e:
                logger.error(f"❌ Lỗi khi xóa backup cũ {file_path}: {e}")