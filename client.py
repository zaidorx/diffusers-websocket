import asyncio
import random
import websockets
import json

async def test():
    async with websockets.connect('ws://localhost:8000', ping_interval=None) as websocket:
        loops = total_num_images // num_images_per_request 
        for i in range(0,loops):
            data= {
            "prompt" : text_prompt,
            "init_image" : "",
            "seed"       : random.randint(1,1000000000),
            "strength"   : 0.75,
            "g_scale"    : 7.5,
            "output_dir" : out_dir,
            "num_images" : num_images_per_request,
            }
            payload = json.dumps(data)
            print(f"Sending data: {payload} to websocket")
            await websocket.send(payload)        
            print("Waiting for result.")
            response = await websocket.recv()
            print(response)

if __name__ == '__main__':
    text_prompt = "cuban cityscape in the style of wifredo lam, trending in artstation, 8k, HQ"
    out_dir = "C:/Users/zaido/Pictures/StableDiffusion/landscapes/"
    num_images_per_request = 4
    total_num_images = 8
    seed = 712112
    num_steps=50
    random.seed(seed)
    asyncio.run(test())