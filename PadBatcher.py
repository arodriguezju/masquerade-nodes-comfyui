import torch
import numpy as np

import numpy as np
import torch.nn.functional as F

def pad_and_batch_images(tensors):
    """
    Receives an array of tensors of shape HWC and float32 type.
    Returns a batched tensor of shape BHWC and float32 type.
    Pads smaller images with -1 values to match the size of the largest image.
    """
    # Find the maximum height and width of the images
    max_height = max(tensor.shape[0] for tensor in tensors)
    max_width = max(tensor.shape[1] for tensor in tensors)

    # Create a list to store the padded tensors
    padded_tensors = []

    # Iterate over the input tensors
    for tensor in tensors:
        # Calculate the padding needed for this tensor
        height_pad = max_height - tensor.shape[0]
        width_pad = max_width - tensor.shape[1]

        # Pad the tensor with -1 values
        padded_tensor = F.pad(tensor, (0, width_pad, 0, height_pad), value=torch.nan)
        # Add the padded tensor to the list
        padded_tensors.append(padded_tensor)

    # Convert the list of padded tensors to a batched tensor
    batched_tensor = torch.stack(padded_tensors, axis=0)

    return batched_tensor

def unpad_image(image):
    # Assuming the tensor format is (H, W, C) and padding value is -1
    assert image.dim() == 3, "Input tensor must have 3 dimensions (H, W, C)"
    
    # Identify rows and columns that contain only -1 values across all channels
    valid_rows = ~(image == -1).any(dim=2).any(dim=1)  # Check each row
    valid_cols = ~(image == -1).any(dim=2).any(dim=0)  # Check each column

    # Find the first and last valid rows and columns
    first_valid_row = valid_rows.nonzero()[0][0]
    last_valid_row = valid_rows.nonzero()[-1][0]
    first_valid_col = valid_cols.nonzero()[0][0]
    last_valid_col = valid_cols.nonzero()[-1][0]

    # Crop the image to these valid rows and columns
    cropped_img = image[first_valid_row:last_valid_row+1, first_valid_col:last_valid_col+1, :]

    return cropped_img

# Example usage
# images = [np.random.rand(32, 32, 3), np.random.rand(28, 28, 3)]  # list of HWC numpy arrays
# batched_images = pad_and_batch_images(images)
# print(batched_images.shape)  # Should print something like torch.Size([2, 3, 32, 32])
