import torch
import numpy as np

def pad_and_batch_images(images):
    # Determine the maximum width and height
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    num_channels = images[0].shape[2]  # Assuming all images have the same number of channels

    # Create a batch tensor with padding value -1
    batch_tensor = torch.full((len(images), num_channels, max_height, max_width), -1)

    # Pad each image and copy it into the batch tensor
    for i, img in enumerate(images):
        img_tensor = img.permute(2, 0, 1)  # Rearrange dimensions from HWC to CHW
        # Calculate padding sizes
        padding_left = 0
        padding_right = max_width - img.shape[1]
        padding_top = 0
        padding_bottom = max_height - img.shape[0]
        # Apply padding
        padded_img = torch.nn.functional.pad(img_tensor, (padding_left, padding_right, padding_top, padding_bottom), value=-1)
        batch_tensor[i] = padded_img

    return batch_tensor

def unpad_image(image):
    # Assuming the tensor format is (H, W, C) and padding value is -1
    assert image.dim() == 3, "Input tensor must have 3 dimensions (H, W, C)"
    
    # Identify rows and columns that contain only -1 values across all channels
    valid_rows = ~(image == -1).all(dim=1).all(dim=1)  # Check each row
    valid_cols = ~(image == -1).all(dim=0).all(dim=0)  # Check each column

    # Crop the image to these valid rows and columns
    cropped_img = image[valid_rows][:, valid_cols]

    return cropped_img

# Example usage
# images = [np.random.rand(32, 32, 3), np.random.rand(28, 28, 3)]  # list of HWC numpy arrays
# batched_images = pad_and_batch_images(images)
# print(batched_images.shape)  # Should print something like torch.Size([2, 3, 32, 32])
