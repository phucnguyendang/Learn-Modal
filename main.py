import modal
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Tạo image với các dependencies cần thiết
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "xformers",
    "Pillow",
    "fastapi",
    "pydantic",
    "numpy",
)

# Tạo app (ứng dụng Modal)
app = modal.App("stable-diffusion-app", image=image)

# Volume để cache model
model_volume = modal.Volume.from_name("sd-models", create_if_missing=True)

# Pydantic models cho request/response
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 512
    height: Optional[int] = 512
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = -1

class GenerateResponse(BaseModel):
    success: bool
    generation_time: float
    image: str
    

@app.cls(
    gpu="L40S",  
    scaledown_window=300, 
    volumes={"/models": model_volume},
)
@modal.concurrent(max_inputs=5)  
class StableDiffusionModel:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import StableDiffusionPipeline
        
        # Đường dẫn cache model
        cache_dir = "/models"
        
        # Load model
        model_id = "runwayml/stable-diffusion-v1-5"
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()
        
        print("Model loaded successfully!")
    
    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = -1
    ) -> tuple[str, float]:
        import torch
        import time
        from PIL import Image
        
        start_time = time.time()
        
        # Set seed nếu được chỉ định
        if seed != -1:
            torch.manual_seed(seed)
        
        # Generate image
        with torch.autocast("cuda"):
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        generation_time = time.time() - start_time
        
        return img_base64, generation_time

# Tạo FastAPI app
web_app = FastAPI(
    title="Stable Diffusion API",
    description="Text-to-Image Generation API using Stable Diffusion",
    version="1.0.0"
)

@web_app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """
    Generate image from text prompt
    """
    try:
        model = StableDiffusionModel()
        img_base64, generation_time = model.generate.remote(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        
        return GenerateResponse(
            success=True,
            generation_time=generation_time,
            image=img_base64,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    app.deploy()