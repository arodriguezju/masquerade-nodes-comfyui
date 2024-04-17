
import torch
import torchvision.transforms.functional as TF
import random


class TxtPromptLoaderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),         
                "file_path": ("STRING", {"multiline": True}),                  
            },
        }

    RETURN_TYPES = ("STRING","STRING", )
    FUNCTION = "run"

    CATEGORY = "PromptLoader"

    def run(self, seed, file_path):
        line = self.get_line_from_file(seed, file_path)
        prompt = line.replace(" ", "_")
        return (line,prompt, )
    

    def get_line_from_file(self, line_number, file_path):
        try:
            with open(file_path, 'r') as file:
                for current_line_number, line in enumerate(file, start=1):
                    if current_line_number == line_number:
                        return line.strip()
            return "Line number out of range."
        except FileNotFoundError:
            return "File not found."


NODE_CLASS_MAPPINGS = {
    "Txt Prompt Loader": TxtPromptLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Txt Prompt Loader": "Txt Prompt Loader",
}