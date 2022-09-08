from helpers.create import StableDiffusionHelper
import random

#Create image from prompt
if __name__ == '__main__':
    prompt = "one red rose laying on the floor, everything else in black and white, fluid art, decor, digital art, wall art, canvas, photorealistic, 8k, trending in artstation"
    out_dir = "C:/Users/zaido/Pictures/StableDiffusion/wall_art/"
    num_images = 10
    seed = 79123
    num_steps=50
    #seed = 168582803
    sd = StableDiffusionHelper(output_dir = out_dir, num_images = num_images, seed = seed, num_steps = num_steps)  
    #sd.process_txt2img(prompt)
    random.seed(seed)  
    for i in range(0,100):        
        sd.process_txt2img(prompt, seed=random.randint(1,1000000000))