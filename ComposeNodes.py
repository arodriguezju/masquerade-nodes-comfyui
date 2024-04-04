import torch
import torchvision.transforms.functional as TF
import random


class ComposeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_batch": ("IMAGE",),             
                "image_batch": ("IMAGE",),  
                "maxSize": ("INT", {"default": 900, "min": 0, "max": 2000, "step": 1}),
                "padding": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "numberOfRepetitions": ("INT", {"default": 5, "min": 0, "max": 5, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE" )
    FUNCTION = "run"

    CATEGORY = "Compose"

    def run(self, mask_batch, image_batch, maxSize, padding, numberOfRepetitions):
        masks, images = self.create_collage(mask_batch, image_batch, maxSize, padding, numberOfRepetitions)
        return (masks, images, )
    

    def create_collage(self, mask, image, maxSize, padding, numberOfRepetitions):
        print("shapes")
        print(mask.shape, image.shape)
        device = torch.device("cpu")
        mask = mask.to(device)
        image = image.to(device)

        p_image = image.permute(0, 3, 1, 2)  # Permute channel because torch expects C, H, W
        p_mask = mask.unsqueeze(1)  # Add a channel dimension to the mask
        B, C, H, W = p_image.shape

        print("shapes")
        print(p_mask.shape, p_image.shape)

        scale_factor = min(maxSize / max(H, W), 1)  # Ensure not to upscale
        newH, newW = int(H * scale_factor), int(W * scale_factor)
        max_separation = int(padding)  # Maximum separation between images
        min_separation = int(padding / 2)  # Minimum separation between images
        base_height = padding * 2 + newH
        base_width = padding * 2 + newW  * numberOfRepetitions + max_separation * (numberOfRepetitions - 1) 

        #Make square
        base_height = max(base_height, base_width)
        base_width = base_height

        
        # Initialize an empty tensor for the base_images with the desired dimensions
        base_images = torch.empty((B, C, base_height, base_width), dtype=torch.float32)

    # Loop over all images in the batch and resize them
        for b in range(B):
        # Resize each image to the new base dimensions
            base_images[b] = TF.resize(p_image[b], size=(base_height, base_width))

        base_masks = torch.zeros((B, 1, base_height, base_width), dtype=torch.float32).to(device)

        y_offset = base_height // 2 - newH // 2

        for b in range(B):
            
            current_x_offset = padding  # Start with padding as the initial x_offset
            for _ in range(numberOfRepetitions):
                # Calculate x_offset for the next image
                x_offset = current_x_offset
                # Update current_x_offset for the next iteration to ensure no overlap
                current_x_offset += newW + random.randint(min_separation, max_separation)  # Adjust spacing between images
                
                # Assign the resized images/masks to the base image/mask at the calculated x_offset
                resized_image = TF.resize(p_image[b], [newH, newW])
                resized_mask = TF.resize(p_mask[b], [newH, newW])

                print("shapes")
                print(resized_image.shape,p_image[b].shape)
                base_images[b, :, y_offset:y_offset+newH, x_offset:x_offset+newW] = resized_image
                base_masks[b, :, y_offset:y_offset+newH, x_offset:x_offset+newW] = resized_mask

        print(base_masks.squeeze(1).shape, base_images.shape)
        return base_masks.squeeze(1), base_images.permute(0,2,3,1)  # Remove the single channel dim from masks for consistency


NODE_CLASS_MAPPINGS = {
    "Compose Image": ComposeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Compose Image": "Compose Image",
}