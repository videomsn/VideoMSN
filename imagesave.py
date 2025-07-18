
"""

import os
import torch
import numpy as np
from PIL import Image
from datetime import datetime

def save_image(image_tensor, folder_path, file_name_prefix="saved_image"):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Function to save a single image
    def save_single_image(image_tensor, folder_path, file_name_prefix):
        # Convert the tensor to a numpy array (C, H, W) -> (H, W, C) for PIL
        image_np = image_tensor.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

        # Check if the image shape is valid
        if image_np.shape[-1] == 3:  # Ensure it has three color channels
            # Convert the numpy array to a PIL image
            image_pil = Image.fromarray(image_np.astype('uint8'))  # Ensure the data type is uint8
            
            # Generate a unique filename using a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{file_name_prefix}_{timestamp}.png"
            
            # Save the image
            image_path = os.path.join(folder_path, file_name)
            image_pil.save(image_path)
        else:
            raise ValueError(f"Invalid image shape: {image_np.shape}")

    # Check if the input is a batch of images
    if len(image_tensor.shape) == 4:  # Batch of images [B, C, H, W]
        batch_size = image_tensor.shape[0]
        for i in range(batch_size):
            save_single_image(image_tensor[i], folder_path, f"{file_name_prefix}_{i}")
    else:  # Single image [C, H, W]
        save_single_image(image_tensor, folder_path, file_name_prefix)

        """
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


def save_image(tensor, path):
    """
    Saves each image in the tensor to the specified folder.
    
    Parameters:
        tensor (torch.Tensor): The tensor containing the images. Shape: [18, 3, 672, 672]
        path (str): The folder where images will be saved.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    # for i in range(tensor.shape[0]):
    #     print("tensor.shape",tensor.shape)
    #     img_array = tensor[i].cpu().permute(1, 2, 0).numpy()  # Convert from [3, 672, 672] to [672, 672, 3] and move to CPU
    #     img = Image.fromarray(np.uint8(img_array))
    #     img.save(os.path.join(path, f'{i}.png'))
        
    for i in range(tensor.shape[0]):
        to_pil = transforms.ToPILImage()
        image = to_pil(tensor[i])
        image.save(os.path.join(path, f'{i}.png'))

# Example usage
# tensor = torch.rand(18, 3, 672, 672) * 255  # Example tensor
# save_image(tensor, '/path/to/save/folder')
