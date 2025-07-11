# Deploy SD1.5 on Modal
---

## 1. Khởi tạo Model

### **Deploy API lên Modal:**
    ```bash
    python -m modal deploy main.py
    ```
### **Lấy URL endpoint** mà Modal cung cấp, ví dụ:
    ```
    https://your-org--stable-diffusion-app-fastapi-app.modal.run/generate
    ```
### **Kiểm tra API docs:**
    - Mở trình duyệt và truy cập: `https://<your-url>/docs`
    - Có thể thử gửi request trực tiếp tại đây.

---

## 2. Hướng dẫn chạy Frontend (Gradio)
### Các bước sử dụng
1. **Cập nhật URL backend**
    - Mở file `gradio_app.py`
    - Thay giá trị biến `BACKEND_API_URL` thành URL `/generate` của bạn, ví dụ:
      ```python
      BACKEND_API_URL = "https://your-org--stable-diffusion-app-fastapi-app.modal.run/generate"
      ```
2. **Chạy giao diện Gradio:**
    ```bash
    python gradio_app.py
    ```
