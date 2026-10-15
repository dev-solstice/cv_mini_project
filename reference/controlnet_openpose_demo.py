# 기존 방식: 사람 사진 -> OpenPose 스켈레톤 -> ControlNet 이미지 생성
# pip install diffusers transformers accelerate controlnet_aux
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
image = Image.open("input.png").convert("RGB")
pose = openpose(image)  # 졸라맨을 넣으면 관절 검출이 안 되는지 확인
pose.save("pose.png")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, torch_dtype=torch.float16).to("cuda")

result = pipe("anime girl, school uniform, white background",
              image=pose, num_inference_steps=25).images[0]
result.save("result.png")