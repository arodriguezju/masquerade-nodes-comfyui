import torch
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms
import numpy as np
from .PadBatcher import unpad_image

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
                "item_size_percentage": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.1}),
                "numberOfRepetitions": ("INT", {"default": 5, "min": 0, "max": 5, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE" )
    FUNCTION = "run"

    CATEGORY = "Compose"

    def run(self, mask_batch, image_batch, maxSize, item_size_percentage, numberOfRepetitions):
        # masks, images = self.create_collage(mask_batch, image_batch, maxSize, item_size_percentage, numberOfRepetitions)
        images = self.create_collage(image_batch, maxSize, item_size_percentage, numberOfRepetitions)

        return (mask_batch, images, )
    


    def create_collage(self, image_batch, max_size, object_size_proportion, number_of_repetitions):
        """
        Create a collage with the object in the image batch.

        Args:
            image_batch: BHWC tensor, float32 type
            max_size: int, maximum size of the output image
            object_size_proportion: float, proportion of the area that the object should occupy
            number_of_repetitions: int, number of times to repeat the object in the collage

        Returns:
            collage_batch: BHWC tensor, float32 type, with the collage images
        """
        batch_size, _, _, channels = image_batch.shape
        collage_batch = torch.zeros((batch_size, max_size, max_size, channels), dtype=torch.float32)

        for i in range(batch_size):
            image = image_batch[i]
            image = unpad_image(image)
            # Resize the image to max_size x max_size
            print("Input Image shape: ", str(image.shape))

            pil_image = transforms.ToPILImage()(image)
            #resize expect CxHxW
            backgorund_image = transforms.Resize((max_size, max_size))(pil_image)
            background_tensor = transforms.ToTensor()(backgorund_image)
            print("Background image shape: ", str(background_tensor.shape))

            # Calculate the scale factor to achieve the desired object size proportion
            height, width = image.shape[1], image.shape[2]
            object_area = height * width
            desired_object_area = max_size ** 2 * object_size_proportion
            scale_factor = np.sqrt(desired_object_area / object_area)

            # Resize the image to the desired object size
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)

            object_image = transforms.Resize((new_height, new_width))(pil_image)
            object_image_tensor = transforms.ToTensor()(object_image)

            print("Object image after resize shape: ", str(background_tensor.shape))

            # Paste the image in the center of the output image
            x_start = (max_size - new_width) // 2
            y_start = (max_size - new_height) // 2
            collage_batch[i, y_start:y_start+new_height, x_start:x_start+new_width, :] = object_image_tensor

            # # Repeat the object in the collage
            # for _ in range(number_of_repetitions - 1):
            #     x_offset = np.random.randint(0, max_size - new_width)
            #     y_offset = np.random.randint(0, max_size - new_height)
            #     collage_batch[i, y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] += image

        return collage_batch

    def create_collage(self, mask, image, maxSize, item_size_percentage, numberOfRepetitions):
        print("shapes")
        print(mask.shape, image.shape)
        device = torch.device("cpu")
        mask = mask.to(device)
        image = image.to(device)

        # p_image = image.permute(0, 3, 1, 2)  # Permute channel because torch expects C, H, W
        # p_mask = mask.unsqueeze(1)  # Add a channel dimension to the mask
        B, _, _, C = image.shape


        base_images = torch.empty((B, maxSize, maxSize, C), dtype=torch.float32)

        print("shapes")
        print(image.shape)
        for b in range(B):
            # H, W, C = image[b].shape
            # scale_factor = min(maxSize / max(H, W), 1)  # Ensure not to upscale

            # newH, newW = int(H * scale_factor), int(W * scale_factor)
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