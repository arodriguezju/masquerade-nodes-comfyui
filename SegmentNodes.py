

from .GroundingDINO import load_groundingdino_model, groundingdino_predict, draw_box_on_image, get_torch_device
from .MaskNodes import tensor2rgba, tensor2rgb
from torchvision.transforms.functional import to_tensor, to_pil_image
from transformers import SamModel, SamProcessor
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from .PadBatcher import pad_and_batch_images, unpad_image
class SegmentNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "box_class": ("STRING", {"multiline": False, "default": "earring"}),
                "box_threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "padding_proportion": ("FLOAT", {"default": 0.4, "min": 0, "max": 1, "step": 0.01}),
                "sam_model": ("STRING", {"multiline": False, "default": "crom87/segmentation_test2"}),
                "sam_base_model": ("STRING", {"multiline": False, "default": "facebook/sam-vit-base"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", )
    FUNCTION = "detect"

    CATEGORY = "Grounding Dino"

    def detect(self, image_batch, box_class, box_threshold, padding_proportion, sam_model, sam_base_model):
        original_image_with_box, crop_image_with_box, crop, box = self.detect_box("GroundingDINO_SwinT_OGC (694MB)", image_batch, box_class, box_threshold, padding_proportion)
        masks = self.segment(sam_model, sam_base_model, crop, box)
        masked_image = self.apply_mask_to_image(masks, crop, 0.1)
        return (masks, crop, masked_image, )
        # draw_box_on_image(crop, torch_box.numpy()).show()

    def detect_box(self, grounding_dino_model_name, image_batch, segmentation_class, threshold, padding_proportion):

        def extract_biggest_box(boxes):
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            biggest_box_idx = torch.argmax(box_areas)
            return boxes[biggest_box_idx]
        
        grounding_dino_model = load_groundingdino_model(grounding_dino_model_name)

        original_images_with_box = []
        cropped_images_with_box = []
        boxes = []
        images = []
        print(image_batch.shape)
        for i in range(image_batch.size(0)):
            tensor_img = image_batch[i]
            print(tensor_img.shape)
            pil_image = Image.fromarray((tensor_img.numpy() * 255).astype(np.uint8))
            # image = Image.fromarray(image.numpy().astype(np.uint8))
            detected_boxes = groundingdino_predict(grounding_dino_model, pil_image, segmentation_class, threshold)
            print(f"Detected {detected_boxes.size(dim=0)} boxes.")
            biggest_box = extract_biggest_box(detected_boxes).numpy()
            original_image = draw_box_on_image(pil_image, biggest_box)
            image, box = self.crop_image_proportional_padding(pil_image, biggest_box, padding_proportion)
            cropped_image = draw_box_on_image(image, box)

            original_images_with_box.append(to_tensor(original_image).permute(1, 2, 0)) #output needs to be [h, w, c]
            cropped_images_with_box.append(to_tensor(cropped_image).permute(1, 2, 0))
            images.append(to_tensor(image).permute(1, 2, 0))
            boxes.append(torch.tensor(box))

        return pad_and_batch_images(original_images_with_box), pad_and_batch_images(cropped_images_with_box), pad_and_batch_images(images), torch.stack(boxes)

    def segment(self, sam_model, sam_model_base, image_batch, box_batch):
        #facebook/sam-vit-huge
        model = SamModel.from_pretrained(sam_model).to(get_torch_device())
        processor = SamProcessor.from_pretrained(sam_model_base)
        model.eval()

        output_masks = []

        for i in range(image_batch.size(0)):
            box = box_batch[i]
            image_tensor = unpad_image(image_batch[i])
            pil_image = Image.fromarray((image_tensor.numpy() * 255).astype(np.uint8))
            image = pil_image.convert("RGB")
            # image.show()
            inputs = processor(image, input_boxes=[[box.tolist()]], return_tensors="pt").to(get_torch_device())
 
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)
           
            masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu(), binarize=False)
            medsam_seg_prob = torch.sigmoid(masks[0])
            # print(masks[0])
            # Image.fromarray(medsam_seg_prob.squeeze(0).squeeze(0).numpy() * 255).show()
            # print(medsam_seg_prob.shape)

            squeezed_tensor = medsam_seg_prob.squeeze(0).squeeze(0)  # Now shape [h, w]
           
            # print(medsam_seg_prob_t.shape)
            # Image.fromarray(medsam_seg_prob_t.numpy()).show()
            output_masks.append(squeezed_tensor)

        return torch.stack(output_masks)


    def crop_image_proportional_padding(self, image, bounding_box, padding_proportion):
       
        x1, y1, x2, y2 = bounding_box
        
        
        # Calculate the bounding box width and height
        bb_width = x2 - x1
        bb_height = y2 - y1
        
        # Calculate the total width and height the padded bounding box should cover
        padding_width = bb_width * padding_proportion
        padding_height = bb_height * padding_proportion
        
       
        
        # Calculate new bounding box with proportional padding
        x1_padded = max(0, x1 - padding_width / 2)
        y1_padded = max(0, y1 - padding_height / 2)
        x2_padded = min(image.width, x2 + padding_width / 2)
        y2_padded = min(image.height, y2 + padding_height / 2)
        
        # Crop the image
        cropped_image = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))

        # Calculate the original bounding box coordinates in the new cropped image space
        new_x1 = x1 - x1_padded
        new_y1 = y1 - y1_padded
        new_x2 = new_x1 + bb_width
        new_y2 = new_y1 + bb_height
        
        return cropped_image, (new_x1, new_y1, new_x2, new_y2)
    
    def apply_mask_to_image(self, masks, image, threshold):
       
        binarized_masks = (masks > threshold).float()        
      
        # Unsqueeze the mask tensor to add a channel dimension, making it [B, H, W, 1]
        binarized_masks = binarized_masks.unsqueeze(-1)
        
        
        # Expand the mask to match the image's shape [B, H, W, C] for broadcasting
        masked_image = image * binarized_masks.expand_as(image)
        
        return masked_image

class GreyScaleToRGBNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),                
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "run"

    CATEGORY = "Grounding Dino"

    def run(self, image_batch):
        print(image_batch.shape)
        rgb =  image_batch.unsqueeze(1).repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        print(rgb.shape)

        return (rgb,  )
        # draw_box_on_image(crop, torch_box.numpy()).show()

    
                               
# Example usage
# image_path = "path_to_your_image.jpg"
# bounding_box = (100, 150, 200, 250)  # Example bounding box
# padding_proportion = 0.1  # Example proportional padding

# cropped_image = crop_image_proportional_padding(image_path, bounding_box, padding_proportion)
# cropped_image.show()


NODE_CLASS_MAPPINGS = {
    "Segment Image": SegmentNode,
    "GreyScaleToRGB": GreyScaleToRGBNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Segment Image": "Segment Image",
    "GreyScaleToRGB": "GreyScaleToRGB",
}

# image = Image.open("crop3_square_1024.png").convert("RGB")
# transform = transforms.ToTensor()
# tensor_image = transform(image)
# batched_image = tensor_image.permute(1, 2, 0).unsqueeze(0)
# node = SegmentNode()
# mask, crop, test,  = node.detect(batched_image, "earring", 0.5, "crom87/segmentation_test2", "facebook/sam-vit-base")
# to_pil_image(mask[0]).show()

# from ComposeNodes import create_collage

# print(mask.shape, crop.shape)

# mask_r, image_r = create_collage(mask, crop, 1000, 100, 2)
# to_pil_image(image_r[0]).show()
# to_pil_image(mask_r[0]).show()


# Example usage
# mask = torch.randn(1, 100, 100)  # Example mask
# image = torch.randn(1, 100, 100, 3)  # Example image
# maxSize = 50
# padding = 10
# numberOfRepetitions = 5
# collage_mask, collage_image = create_collage(mask, image, maxSize, padding, numberOfRepetitions)
# print(collage_mask.shape, collage_image.shape)
