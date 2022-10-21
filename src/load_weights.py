import os

from diffusers import DiffusionPipeline
import torch
from dotenv import load_dotenv


load_dotenv()
auth_token = os.getenv("API_TOKEN") or True

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        # torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=auth_token,
    )
