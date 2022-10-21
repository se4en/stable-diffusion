# src: https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting/blob/main/app.py
from typing import Any, Tuple
import os

from dotenv import load_dotenv
import gradio as gr
import torch
from torchvision import transforms
from diffusers import DiffusionPipeline


load_dotenv()
auth_token = os.getenv("API_TOKEN") or True

device = "cuda" if torch.cuda.is_available() else "cpu"


def dummy_checker(image: Any, *args, **kwargs) -> Tuple[Any, bool]:
    return image, False


pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    # torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token=auth_token,
).to(device)

pipe.safety_checker = dummy_checker

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ]
)


def read_content(file_path: str) -> str:
    """read the content of target file"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return content


def predict(dict, prompt=""):
    init_image = dict["image"].convert("RGB").resize((512, 512))
    mask = dict["mask"].convert("RGB").resize((512, 512))
    output = pipe(prompt=prompt, image=init_image, mask_image=mask, guidance_scale=7.5)
    return output.images[0]


css = """
.container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
"""

image_blocks = gr.Blocks(css=css)

with image_blocks as demo:
    gr.HTML(read_content("src/header.html"))
    with gr.Group():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    image = gr.Image(
                        source="upload",
                        tool="sketch",
                        elem_id="image_upload",
                        type="pil",
                        label="Upload",
                    ).style(height=400)
                    with gr.Row(elem_id="prompt-container").style(
                        # mobile_collapse=False,
                        equal_height=True
                    ):
                        prompt = gr.Textbox(
                            placeholder="Your prompt (what you want in place of what is erased)",
                            show_label=False,
                        )
                        btn = gr.Button("Inpaint!").style(
                            # margin=False,
                            # rounded=(False, True, True, False),
                            full_width=False,
                        )
                with gr.Column():
                    image_out = gr.Image(label="Output").style(height=400)

            btn.click(fn=predict, inputs=[image, prompt], outputs=image_out)

            gr.HTML(
                """
                    <div class="footer">
                        <p>Model by <a href="https://huggingface.co/runwayml" style="text-decoration: underline;" target="_blank">RunwayML</a> - Gradio Demo by 🤗 Hugging Face
                        </p>
                    </div>
                    <div class="acknowledgments">
                        <p><h4>LICENSE</h4>
        The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                        <p><h4>Biases and content acknowledgment</h4>
        Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
                    </div>
                """
            )


if __name__ == "__main__":
    image_blocks.launch(server_name="0.0.0.0")
