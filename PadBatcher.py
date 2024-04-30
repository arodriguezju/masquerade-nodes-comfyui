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

def unpad_image(tensor):
    """
    Unpads a tensor by removing torch.nan values.
    Expects a tensor of shape HWC, float32.
    Returns a tensor of shape HWC, float32 with new size.
    """
    # Find the indices of non-NaN values in each dimension
    height_idx = torch.nonzero(~torch.isnan(tensor[:, :, 0]), as_tuple=True)[0]
    width_idx = torch.nonzero(~torch.isnan(tensor[:, 0, :]), as_tuple=True)[0]

    # Find the minimum and maximum indices in each dimension
    min_height = height_idx.min()
    max_height = height_idx.max() + 1
    min_width = width_idx.min()
    max_width = width_idx.max() + 1

    # Slice the tensor to remove NaN values
    unpadded_tensor = tensor[min_height:max_height, min_width:max_width, :]

    return unpadded_tensor
# Example usage
# images = [np.random.rand(32, 32, 3), np.random.rand(28, 28, 3)]  # list of HWC numpy arrays
# batched_images = pad_and_batch_images(images)
# print(batched_images.shape)  # Should print something like torch.Size([2, 3, 32, 32])
