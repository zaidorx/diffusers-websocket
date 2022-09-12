import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import uuid
import os
from io import BytesIO

class StableDiffusionHelper:
    def __init__(self, prompt = '', img2img = False, num_images = 4, seed = 41, num_steps=50, output_dir = '.'):
        self.num_images = num_images
        self.num_steps = num_steps
        self.seed = seed
        self.out_dir = output_dir
        self.prompt = prompt
        self.height = 512
        self.width = 512
        self.params = ""
        self.img2img = img2img
        self.generator = torch.Generator("cuda").manual_seed(self.seed)
        if img2img:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", 
                revision="fp16", 
                torch_dtype=torch.float16,
                use_auth_token=True
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", 
                revision="fp16", 
                torch_dtype=torch.float16,
                use_auth_token=True
            )
        
        pipe.safety_checker = self.dummy
        self.pipe = pipe.to("cuda")   
        pipe = pipe.enable_attention_slicing()
        
    
    def gen_params_string(self):
        if self.img2img:
            self.params = f"parameters: initial image: {self.init_image}, num_images: {self.num_images}, strength: {self.strength}, self.guidance_scale: {self.guidance_scale}, seed: {self.seed}"
        else:
            self.params = f"parameters: num_images: {self.num_images}, num_steps: {self.num_steps}, seed: {self.seed}"
        self.params = f"{self.params}, width: {self.width}, height: {self.height}"
    
    def dummy(self, images, **kwargs):
        #https://www.reddit.com/r/StableDiffusion/comments/wxba44/disable_hugging_face_nsfw_filter_in_three_step/
        return images, False

    def show_images(self, imgs, rows, cols):
        #assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        grid.show()

    def save_images(self, imgs, create_folder):
        prompts_file = os.path.join(self.out_dir, "prompts.txt")
        outfolder = ''
        if os.path.exists(prompts_file):
            with open(prompts_file, "r") as f:
                prompts = f.readlines()
            for line in prompts:
                if self.prompt in line:
                    outfolder = line[0:36]
        if len(outfolder) == 0:
            outfolder = str(uuid.uuid4())
        sample_path = os.path.join(self.out_dir, outfolder) if create_folder else self.out_dir
        os.makedirs(sample_path, exist_ok=True)
        print(f"saving to folder {sample_path}")
        base_count = len(os.listdir(sample_path))
        if base_count == 0:
            base_count = 1
        filenames = ''
        for image in imgs:
            filename = f"{base_count:05}.png"
            filenames = f"{filenames}{filename}, "
            image.save(os.path.join(sample_path, filename))
            base_count += 1
        filenames = filenames.rstrip(filenames[-3])

        if create_folder:
            promt_text = f"{outfolder} {self.params}"
            with open(prompts_file, "a+") as f:
                f.write(f"{promt_text} prompt: {self.prompt}")
                f.write("\n")

        prompts_file = os.path.join(sample_path, "prompts.txt")
        if os.path.exists(prompts_file) or not create_folder:
            self.gen_params_string()
            with open(prompts_file, "a+") as f:
                if not create_folder:
                    f.write(self.prompt)
                f.write(self.params)
                f.write(f",  files: {filenames}")
                f.write("\n")                
        else:
            self.gen_params_string()
            with open(prompts_file, "a") as f:
                f.write("diffusers:StableDiffusionPipeline:txt2img (script:create.py)\n")
                f.write(self.prompt)
                f.write("\n")
                f.write(sample_path)
                f.write("\n")
                f.write(self.params)
                f.write(f",  files: {filenames}")
                f.write("\n")
        
        

    def process_txt2img(self, init_image = None, prompt = '', seed =-1, num_steps=-1, output_dir='', num_images = -1, strength = 0.75,
                              g_scale = 7.5, height = -1, width = -1, create_unique_folder = True):
        if self.img2img and init_image is None:
            print("Please specify initial image to process")
            return
        if len(prompt) > 0:
            self.prompt = prompt
        if len(self.prompt) == 0:
            print("Please specify a prompt")
            return
        if seed > 0:
            self.seed = seed
            self.generator = torch.Generator("cuda").manual_seed(self.seed)
        if num_steps > 0:
            self.num_steps = num_steps
        if num_images > 0:
            self.num_images = num_images
        if len(output_dir) > 0:
                self.out_dir = output_dir
        if width > 0:
            self.width = width
        if height > 0:
            self.height = height
        print(f"Saving to: {self.out_dir}")        
        images = []
        if self.img2img:
            from_image = Image.open(init_image).convert("RGB")
            from_image = from_image.resize((512, 512))       
            self.strength = strength
            self.guidance_scale = g_scale
            self.init_image = init_image
        count = 0
        for i in range (0, self.num_images):
            with autocast("cuda"):
                if self.img2img:
                    '''
                        prompt: Union[str, List[str]],
                        init_image: Union[torch.FloatTensor, PIL.Image.Image],
                        strength: float = 0.8,
                        num_inference_steps: Optional[int] = 50,
                        guidance_scale: Optional[float] = 7.5,
                        eta: Optional[float] = 0.0,
                        generator: Optional[torch.Generator] = None,
                        output_type: Optional[str] = "pil"
                    '''
                    result = self.pipe(prompt=prompt, init_image=from_image, strength= self.strength, guidance_scale=self.guidance_scale, generator= self.generator)
                else:
                    '''
                    Parametros que se le pueden pasar a pipe!!
                    prompt: Union[str, List[str]],  El Prompt
                    height: Optional[int] = 512,  Altura de la imagen a generar
                    width: Optional[int] = 512,  Ancho de la imagen a generar
                    num_inference_steps: Optional[int] = 50, Numero de steps
                    guidance_scale: Optional[float] = 7.5, 
                    eta: Optional[float] = 0.0,
                    generator: Optional[torch.Generator] = None,
                    latents: Optional[torch.FloatTensor] = None,
                    output_type: Optional[str] = "pil",
                    **kwargs,'''
                    result =self.pipe(self.prompt, num_inference_steps = self.num_steps, generator = self.generator, width=self.width, height=self.height) 
                image = result.images[0]
                images.append(image)
                if len(images) > 1:
                    count += len(images)
                    self.save_images(images, create_unique_folder)
                    images =[]                    
        if len(images) > 0:
            self.save_images(images, create_unique_folder)
            count += len(images)
        return count
        #self.show_images(images, rows=1, cols=self.num_images)


