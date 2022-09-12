import asyncio 
import websockets
import json
from helpers.create import StableDiffusionHelper
import socket
import numpy as np

#https://stackoverflow.com/a/49677241/1694701
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# create handler for each connection
sd = {} 
response = []
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
                                num_images=data['num_images'], height=data['height'], width=data['width'], num_steps=data['num_steps'])
    response = result

async def producer_handler(websocket, path):
    global response
    while True:
        message = await producer()  
        print(f"sending {type(message)}")      
        result = json.dumps(message, cls=NumpyEncoder)
        await websocket.send(result)
        response = []

async def producer():
    while len(response) == 0:
        await asyncio.sleep(1)        
    return response

 
if __name__=="__main__": 
    hostname=socket.gethostname()   
    IPAddr=socket.gethostbyname(hostname)   
    print("Your Computer Name is:"+hostname)   
    print("Your Computer IP Address is:"+IPAddr) 
    print("Starting Stable Difussion server.....")
    sd = StableDiffusionHelper() 
    start_server = websockets.serve(handler, "", 8000, ping_interval=None, max_size=None)    
    asyncio.get_event_loop().run_until_complete(start_server)
    print(f"Server listening at {IPAddr}:8000")
    asyncio.get_event_loop().run_forever()