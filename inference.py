from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

model_dir = "./mm_model" #replace with path to your saved model directory

unet = UNet2DConditionModel.from_pretrained(model_dir+"/unet",
                                            safety_checker = None,
                                            requires_safety_checker = False)

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained(model_dir+"/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, dtype=torch.float16).to("cuda")

prompt = "A portrait of mfm"
num_inference_steps = 90
guidance_scale = 7
image = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
image_title = prompt.lower().replace(" ", "_")+"-"+str(num_inference_steps)+"-"+str(guidance_scale)+".png"
image.save("generated_images/"+image_title+".png")