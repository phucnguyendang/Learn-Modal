import gradio as gr
import requests
import base64
import io
from PIL import Image

# --- CẤU HÌNH ---
BACKEND_API_URL = "https://phucnguyendang--stable-diffusion-app-fastapi-app.modal.run/generate" 

def generate_image_from_api(prompt: str, negative_prompt: str, width: int, height: int, num_inference_steps: int, guidance_scale: float, seed: int):
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed
    }

    try:
        # Gửi request POST đến backend
        print("Đang gửi request đến backend...")
        response = requests.post(BACKEND_API_URL, json=payload, timeout=300) 
        response.raise_for_status() 
        data = response.json()
        img_base64 = data.get("image")
        if not img_base64:
            raise gr.Error("API không trả về ảnh.")
            
        print("Đã nhận được ảnh, đang giải mã...")
        image_data = base64.b64decode(img_base64)
        pil_image = Image.open(io.BytesIO(image_data))
        
        return pil_image

    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gọi API: {e}")
        # Cố gắng đọc lỗi chi tiết từ response của FastAPI
        try:
            error_detail = e.response.json().get('detail', str(e))
        except:
            error_detail = str(e)
        raise gr.Error(f"Lỗi kết nối đến backend: {error_detail}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        raise gr.Error(f"Đã có lỗi xảy ra: {e}")


# Xây dựng giao diện Gradio
gradio_ui = gr.Interface(
    fn=generate_image_from_api,
    inputs=[
        gr.Textbox(label="Prompt", info="Mô tả ảnh bạn muốn tạo"),
        gr.Textbox(label="Negative Prompt", value="blurry, ugly, deformed", info="Những thứ bạn không muốn thấy trong ảnh"),
        gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Width"),
        gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Height"),
        gr.Slider(minimum=10, maximum=50, step=1, value=20, label="Inference Steps"),
        gr.Slider(minimum=1.0, maximum=15.0, step=0.5, value=7.5, label="Guidance Scale"),
        gr.Number(label="Seed", value=-1, precision=0, info="-1 nghĩa là ngẫu nhiên"),
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Stable Diffusion Image Generator ",
)

if __name__ == "__main__":
    gradio_ui.launch(share=True, debug = True)
