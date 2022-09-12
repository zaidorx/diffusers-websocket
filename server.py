import asyncio 
import websockets
import json
from helpers.create import StableDiffusionHelper
 
# create handler for each connection
sd = {} 
response = ''
async def handler(websocket, path):
    consumer_task = asyncio.ensure_future(
        consumer_handler(websocket, path))
    producer_task = asyncio.ensure_future(
        producer_handler(websocket, path))
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

async def consumer_handler(websocket, path):
    async for message in websocket:
        consumer(message)

def consumer(data):
    global response
    data = json.loads(data)
    print()
    print(data)
    print()
    if "message" in data:
        return
    result = sd.process_txt2img(prompt=data['prompt'], init_image = data['init_image'], seed=data['seed'], 
                                strength=data['strength'], g_scale=data['g_scale'], output_dir=data['output_dir'],
                                num_images=data['num_images'], width=data['width'], height=data['height'], create_unique_folder = data['create_unique_folder'])
    if result > 1:
        response = f"{result} images created successfully"
    else:
        response = f"{result} image created successfully"

async def producer_handler(websocket, path):
    global response
    while True:
        message = await producer()
        await websocket.send(message)
        response = ''

async def producer():
    while len(response) == 0:
        await asyncio.sleep(1)
    return response

 
if __name__=="__main__": 
    print("Starting Stable Difussion server.....")
    sd = StableDiffusionHelper() 
    start_server = websockets.serve(handler, "localhost", 8000, ping_interval=None)    
    asyncio.get_event_loop().run_until_complete(start_server)
    print("Server listening at localhost:8000")
    asyncio.get_event_loop().run_forever()