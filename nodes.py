import os
import sys
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)

from PIL import Image

import argparse
import torch
import numpy as np
import tempfile
import gc
from torch.cuda import amp
from typing import List, Tuple, Union

#/home/admin/ComfyUI/models/t5/t5-v1.1-xxl/models--DeepFloyd--t5-v1_1-xxl/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37
#/home/admin/ComfyUI/models/diffusers/PixArt-LCM-XL-2-1024-MS

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]


styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

def flush():
    gc.collect()
    torch.cuda.empty_cache()

class PixArtT5Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t5_path": ("STRING", {"default": "/home/admin/ComfyUI/models/t5/t5-v1.1-xxl/models--DeepFloyd--t5-v1_1-xxl/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37"}),
                "pixart_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/PixArt-LCM-XL-2-1024-MS"}),
            },
        }
        
    RETURN_TYPES = ("PixArtT5",)
    RETURN_NAMES = ("t5",)
    FUNCTION = "run"
    CATEGORY = "PixArt"
    
    def run(self,t5_path,pixart_path):
        from transformers import T5EncoderModel
        from diffusers import PixArtAlphaPipeline

        text_encoder = T5EncoderModel.from_pretrained(
            t5_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

        pipe = PixArtAlphaPipeline.from_pretrained(
            pixart_path,
            text_encoder=text_encoder,
            transformer=None,
            device_map="auto"
        )
        #pipe.to("cpu")

        return (pipe,)

class PixArtT5EncodePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PixArtT5",),
                "style": (STYLE_NAMES,),
                "prompt": ("STRING", {"default": "cute cat"}),
                "negative": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("prompt_embeds","prompt_attention_mask","negative_embeds","negative_prompt_attention_mask",)
    RETURN_NAMES = ("prompt_embeds","prompt_attention_mask","negative_embeds","negative_prompt_attention_mask",)
    FUNCTION = "run"
    CATEGORY = "PixArt"

    def run(self,pipe,style,prompt,negative):
        prompt, negative = apply_style(style, prompt, negative)
        
        pipe.to('cuda')
        with amp.autocast(enabled=True):
            with torch.no_grad():
                prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)

        pipe.to('cpu')
        flush()

        return (prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask,)

class PixArtLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixart_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/PixArt-LCM-XL-2-1024-MS"}),
                "yoso_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/yoso_pixart1024"}),
                "use_yoso": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("PixArt",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "run"
    CATEGORY = "PixArt"

    def run(self,pixart_path,yoso_path,use_yoso):
        from transformers import T5EncoderModel
        from diffusers import PixArtAlphaPipeline, LCMScheduler, Transformer2DModel

        if use_yoso:
            transformer = Transformer2DModel.from_pretrained(
                yoso_path, torch_dtype=torch.float16).to('cuda')

            pipe = PixArtAlphaPipeline.from_pretrained(
                pixart_path,
                transformer=transformer,
                text_encoder=None,
                torch_dtype=torch.float16,
            )
        else:
            pipe = PixArtAlphaPipeline.from_pretrained(
                pixart_path,
                text_encoder=None,
                torch_dtype=torch.float16,
            )
        #pipe.to("cpu")

        return (pipe,)

class PixArtSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PixArt",),
                "prompt_embeds": ("prompt_embeds",),
                "prompt_attention_mask": ("prompt_attention_mask",),
                "negative_embeds": ("negative_embeds",),
                "negative_prompt_attention_mask": ("negative_prompt_attention_mask",),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}), 
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0, "max": 20}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "schduler": (["lcm","dpm"], {"defult": "lcm"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "run"
    CATEGORY = "PixArt"

    def run(self,pipe,prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask,width,height,steps,guidance_scale,seed,schduler):
        from diffusers import PixArtAlphaPipeline, LCMScheduler, Transformer2DModel, DPMSolverMultistepScheduler

        pipe.to("cuda")
        if schduler=='lcm':
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.prediction_type = "v_prediction"
        if schduler=='dpm':
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.prediction_type = "v_prediction"
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with amp.autocast(enabled=True):
            images = pipe(
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                num_images_per_prompt=1,
                output_type="latent",
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images.half()

        return ({"samples":images},)

class PixArtImageDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PixArt",),
                "samples": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "PixArt"

    def run(self,pipe,samples):
        latents=samples["samples"]
        output_image=None

        with amp.autocast(enabled=True):
            with torch.no_grad():
                images_list = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)
                # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
                data = []
                for image in images_list:
                    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

                    data.append(torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0))

        return torch.cat(tuple(data), dim=0).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "PixArtT5Loader":PixArtT5Loader,
    "PixArtT5EncodePrompt":PixArtT5EncodePrompt,
    "PixArtLoader":PixArtLoader,
    "PixArtSampler":PixArtSampler,
    "PixArtImageDecode":PixArtImageDecode,
}
