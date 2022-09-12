import asyncio
import random
import websockets
import json
import time
import os
import uuid


def save_images(imgs, data):
    prompt = data['prompt']
    params = f"parameters: num_images: {data['num_images']}, num_steps: {data['num_steps']}, seed: {data['seed']}"
    params = f"{params}, width: {data['width']}, height: {data['height']}"
    prompts_file = os.path.join(out_dir, "prompts.txt")
    outfolder = ''
    if os.path.exists(prompts_file):
        with open(prompts_file, "r") as f:
            prompts = f.readlines()
        for line in prompts:
            if prompt in line:
                outfolder = line[0:36]
    if len(outfolder) == 0:
        outfolder = str(uuid.uuid4())
    sample_path = os.path.join(out_dir, outfolder)
    os.makedirs(sample_path, exist_ok=True)
    print(f"saving to folder {sample_path}")
    base_count = len(os.listdir(sample_path))
    filenames = ''
    for image in imgs:
        filename = f"{base_count:05}.png"
        filenames = f"{filenames}{filename}, "
        image.save(os.path.join(sample_path, filename))
        base_count += 1
    filenames = filenames.rstrip(filenames[-3])

    promt_text = f"{outfolder} {params}"
    with open(prompts_file, "a+") as f:
        f.write(f"{promt_text} prompt: {prompt}")
        f.write("\n")

    prompts_file = os.path.join(sample_path, "prompts.txt")
    if os.path.exists(prompts_file):
        with open(prompts_file, "a+") as f:
            f.write(params)
            f.write(f",  files: {filenames}")
            f.write("\n")                
    else:
        with open(prompts_file, "a") as f:
            f.write("diffusers:StableDiffusionPipeline:txt2img (script:create.py)\n")
            f.write(prompt)
            f.write("\n")
            f.write(sample_path)
            f.write("\n")
            f.write(params)
            f.write(f",  files: {filenames}")
            f.write("\n")
        
        
async def test():
    async with websockets.connect('ws://192.168.0.201:8000', ping_interval=None, max_size=None) as websocket:
        loops = total_num_images // num_images_per_request 
        global seed
        seed = random.randint(1,1000000000)
        for i in range(0,loops):
            data= {
            "prompt" : text_prompt,
            "init_image" : "",
            "seed"       : random.randint(1,1000000000),
            "strength"   : 0.75,
            "g_scale"    : 7.5,
            "output_dir" : out_dir,
            "num_images" : num_images_per_request,
            "width"      : 512,
            "height"     : 512,
            "create_unique_folder" : False, #Create a unique folder for each unique prompt or not,
            "num_steps" : num_steps
            }
            payload = json.dumps(data)
            print(f"Sending data: {payload} to websocket")
            await websocket.send(payload)        
            print("Waiting for result.")
            response = await websocket.recv()
            if len(response) > 0:
                save_images(response, data)
            else:
                print('No images were generated')

if __name__ == '__main__':
    prompts = [
        "photorealistic, highly detailed vibrant, ancient greek city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
        "photorealistic, Highly detailed vibrant, ancient roman city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, ancient spanish city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, ancient russian city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
        "photorealistic, highly detailed vibrant, ancient cuban city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
        "photorealistic, highly detailed vibrant, ancient chinese city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, medieval  city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, industrial age city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, dark ages city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, nineteen century city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, roaring twenties city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, modern city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, futuristic city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, cyberpunk city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
		"photorealistic, highly detailed vibrant, scifi city,  beautiful sunny day, lush gardens, street level view, wide lens,   octane render, trending on artstation, 8K, HQ",
    ]
    text_prompt = "Photorealistic image of a futuristic toronto skyline, 8k"
    out_dir = "C:/Users/zaido/Pictures/StableDiffusion/cities/19/"
    num_images_per_request = 1
    total_num_images = 10#len(prompts)
    seed = time.time_ns() // 1000000  #968582803
    num_steps=50
    random.seed(seed)
    asyncio.run(test())