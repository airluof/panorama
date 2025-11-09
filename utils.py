import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForInpainting,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from PIL import Image

# from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler

width, height = 1024, 512


def roll(original_image: np.ndarray, offset=100) -> np.ndarray:
    return np.roll(original_image, (0, offset), axis=(0, 1))


def unroll(original_image: np.ndarray, offset=100) -> np.ndarray:
    return np.roll(original_image, (0, -offset), axis=(0, 1))


def roll_and_mask(image: np.ndarray, mask_offset: np.ndarray) -> np.ndarray:
    mask = np.array(image)
    mask[:, :mask_offset, :] = 255
    mask[:, -mask_offset:, :] = 255
    mask[:, mask_offset:-mask_offset, :] = 0

    rolled_mask = roll(mask, 2 * mask_offset)
    rolled_image = roll(image, 2 * mask_offset)
    return rolled_image, rolled_mask


def generate_w_depth(prompt: str, depth_image: Image) -> Image:
    # prompt="hdri view, an indoor view of a room as a 360 panorama from the center of the room, in the style of <s0><s1"
    generator = torch.Generator("cuda").manual_seed(1024)

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    ).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.enable_model_cpu_offload()

    # depth_image=Image.open('pano_depth.png')

    # generate image
    generated_image = pipe(
        prompt, image=depth_image, width=width, height=height
    ).images[0]

    return generated_image


def generate_wo_depth(prompt: str) -> Image:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    pipe.load_lora_weights("jbilcke-hf/sdxl-panorama", weight_name="lora.safetensors")

    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

    embedding_path = hf_hub_download(
        repo_id="jbilcke-hf/sdxl-panorama", filename="embeddings.pti", repo_type="model"
    )
    embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
    embhandler.load_embeddings(embedding_path)

    generator = torch.Generator("cuda").manual_seed(1024)

    generated_image = pipe(
        prompt,
        height=height,
        width=width,
        cross_attention_kwargs={"scale": 0.8},
        generator=generator,
    ).images[0]

    return generated_image


## Inpaint an image to address the edge mismatch
def inpaint_image(image: Image, mask: Image, seed: int = 1024) -> Image:

    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda:1")

    prompt = ""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipe.enable_model_cpu_offload()

    inpainted_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=10,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
        guidance_scale=1,  # don't use prompt, just inpaint
        width=1024,
        height=512,
    ).images[0]

    return inpainted_image


def inpaint_generated_image(image: Image, offset: int = 100) -> Image:
    rolled_image, rolled_mask = roll_and_mask(np.array(image), offset)
    inpainted_image = inpaint_image(
        Image.fromarray(rolled_image), Image.fromarray(rolled_mask)
    )

    return np.array(unroll(inpainted_image, offset))
