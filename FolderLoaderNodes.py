
import torch
import torchvision.transforms.functional as TF
import random
import os
from PIL import Image
import numpy as np

class ImageFromFolderLoaderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),   
                "folder_path": ("STRING", {"multiline": True}),                  
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "run"

    CATEGORY = "PromptLoader"

    def run(self, seed, folder_path):
        image = self.get_line_from_file(seed, folder_path)
        return (image, )
    

    def load_images_from_folder(self, index, folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        files.sort()  # Sort files for consistent ordering
        file = files[index]
        
        image = Image.open(file).convert('RGB')
        img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        # add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.permute(0, 3, 1, 2)        


NODE_CLASS_MAPPINGS = {
    "Image from folder loader": ImageFromFolderLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image from folder loader": "Image from folder loader",
}