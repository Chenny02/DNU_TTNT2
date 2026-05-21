# ================================================
# SNAKE LABELER - Dán nhãn tiếng Việt cho ảnh rắn
# Sử dụng Google Gemini Vision AI
# ================================================
import os, json, base64
import google.generativeai as genai
from PIL import Image

# ---- ĐỌC API KEY ----
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_key_here')
genai.configure(api_key=GEMINI_API_KEY)

# ---- CẤU HÌNH ----
IMG_FOLDER  = './snake_images'
OUTPUT_FILE = 'snake_labels.json'
model       = genai.GenerativeModel('gemini-1.5-flash')

# ---- PROMPT TIẾNG VIỆT ----
PROMPT = """
Bạn là chuyên gia nhận diện rắn. Hãy phân tích ảnh này và trả lời theo định dạng JSON sau:
{
  "ten_loai": "tên loài rắn bằng tiếng Việt (ví dụ: Rắn hổ mang chúa)",
  "ten_khoa_hoc": "tên khoa học (nếu biết)",
  "mau_sac": "mô tả màu sắc chính của rắn",
  "dac_diem": "đặc điểm nổi bật dễ nhận biết",
  "muc_do_nguy_hiem": "Không nguy hiểm / Thấp / Trung bình / Cao / Rất cao",
  "co_doc": true hoặc false,
  "mo_ta": "mô tả ngắn gọn 1-2 câu bằng tiếng Việt",
  "co_ran": true hoặc false (có thấy rắn trong ảnh không)
}
CHỈ trả về JSON, không giải thích thêm.
"""

def encode_image(path):
    """Đọc ảnh và encode base64"""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_extension(path):
    ext = os.path.splitext(path)[1].lower()
    return {'jpg': 'image/jpeg', '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', '.png': 'image/png'}.get(ext, 'image/jpeg')

def label_image(path):
    """Gửi ảnh lên Gemini và nhận nhãn tiếng Việt"""
    try:
        img = Image.open(path).convert('RGB')
        # Resize nếu ảnh quá lớn (tiết kiệm quota)
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
            temp_path = path + '_temp.jpg'
            img.save(temp_path, 'JPEG')
            path = temp_path

        img_data = {
            'mime_type': 'image/jpeg',
            'data': encode_image(path)
        }
        response = model.generate_content([PROMPT, img_data])
        text = response.text.strip()

        # Làm sạch markdown code block nếu có
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        text = text.strip('`').strip()

        label = json.loads(text)

        # Xoá file tạm nếu có
        if path.endswith('_temp.jpg') and os.path.exists(path):
            os.remove(path)

        return label

    except json.JSONDecodeError:
        return {'loi': 'Không parse được JSON', 'raw': response.text[:200]}
    except Exception as e:
        return {'loi': str(e)}

def build_labels(folder):
    """Dán nhãn toàn bộ ảnh trong thư mục"""
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not files:
        print(f'Không tìm thấy ảnh nào trong {folder}')
        print('Hãy copy ảnh rắn vào thư mục snake_images/ rồi chạy lại.')
        return {}

    print(f'Tìm thấy {len(files)} ảnh. Bắt đầu dán nhãn...\n')
    results = {}

    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        print(f'[{i+1}/{len(files)}] Đang xử lý: {fname}')
        label = label_image(path)
        results[fname] = label

        # In kết quả ngắn gọn
        if 'loi' not in label:
            print(f'  → {label.get("ten_loai", "?")} | '
                  f'Nguy hiểm: {label.get("muc_do_nguy_hiem", "?")} | '
                  f'Có rắn: {label.get("co_ran", "?")}')
        else:
            print(f'  → Lỗi: {label["loi"]}')

        # Lưu tạm sau mỗi 10 ảnh (tránh mất dữ liệu)
        if (i + 1) % 10 == 0:
            json.dump(results, open(OUTPUT_FILE, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
            print(f'  [Đã lưu tạm {i+1} ảnh]')

    # Lưu kết quả cuối
    json.dump(results, open(OUTPUT_FILE, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'\n✅ Hoàn thành! Đã dán nhãn {len(results)} ảnh.')
    print(f'📄 Kết quả lưu tại: {OUTPUT_FILE}')
    return results

def print_summary(results):
    """In thống kê tổng quan"""
    total     = len(results)
    co_ran    = sum(1 for v in results.values() if v.get('co_ran'))
    co_doc    = sum(1 for v in results.values() if v.get('co_doc'))
    co_loi    = sum(1 for v in results.values() if 'loi' in v)

    print('\n========== THỐNG KÊ ==========')
    print(f'Tổng số ảnh      : {total}')
    print(f'Ảnh có rắn       : {co_ran}')
    print(f'Rắn độc          : {co_doc}')
    print(f'Lỗi xử lý        : {co_loi}')
    print('==============================')

# ---- CHẠY CHƯƠNG TRÌNH ----
if __name__ == '__main__':
    # Load .env thủ công (không cần cài thêm dotenv)
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        for line in open(env_path, encoding='utf-8'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())
        # Cập nhật lại API key sau khi đọc .env
        genai.configure(api_key=os.getenv('GEMINI_API_KEY', GEMINI_API_KEY))

    # Kiểm tra thư mục ảnh
    if not os.path.exists(IMG_FOLDER):
        os.makedirs(IMG_FOLDER)
        print(f'Đã tạo thư mục {IMG_FOLDER}. Vui lòng copy ảnh rắn vào đó.')
    else:
        results = build_labels(IMG_FOLDER)
        if results:
            print_summary(results)
