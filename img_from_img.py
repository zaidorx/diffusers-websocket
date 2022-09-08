from helpers.create import StableDiffusionHelper
import random

#Create image from prompt
if __name__ == '__main__':
    text_prompt = "urban cuban lanscape, wide view, art by Victor Manuel, HQ"
    out_dir = "C:/Users/zaido/Pictures/StableDiffusion/landscapes/"
    num_images = 10
    seed = 79123
    num_steps=50
    #seed = 168582803
    seed_image = ''#"C:/Users/zaido/Pictures/StableDiffusion/wall_art/00088.png"
    sd = StableDiffusionHelper(img2img=False, output_dir = out_dir, num_images = num_images, seed = seed, num_steps = num_steps)  
    #sd.process_txt2img(prompt=text_prompt, init_image = seed_image)
    random.seed(seed)  
    for i in range(0,2):        
        sd.process_txt2img(prompt=text_prompt, init_image = seed_image, seed=random.randint(1,1000000000), strength=random.random(), g_scale=random.randint(1,10))