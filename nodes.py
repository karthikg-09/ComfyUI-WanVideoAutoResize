import torch
import math
import numpy as np

class WanVideoAutoImgResize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
            "generation_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Target width for the image"}),
            "generation_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8, "tooltip": "Target height for the image"}),
            "adjust_resolution": ("BOOLEAN", {"default": True, "tooltip": "Adjusts the resolution to maintain aspect ratio while fitting within target dimensions"})
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, image, generation_width, generation_height, adjust_resolution):
        # Handle different image dimensions
        if len(image.shape) == 3:  # Single image: [H, W, C]
            image = image.unsqueeze(0)  # Convert to [1, H, W, C]
        
        # Original dimensions
        batch_size, original_height, original_width, channels = image.shape
        
        # Calculate target dimensions
        if adjust_resolution:
            # Calculate aspect ratios
            original_aspect = original_width / original_height
            target_aspect = generation_width / generation_height
            
            # Adjust dimensions to maintain aspect ratio
            if original_aspect > target_aspect:  # Image is wider
                # Width is the limiting factor
                new_width = generation_width
                new_height = int(new_width / original_aspect)
                # Make sure height is divisible by 8
                new_height = (new_height // 8) * 8
            else:  # Image is taller
                # Height is the limiting factor
                new_height = generation_height
                new_width = int(new_height * original_aspect)
                # Make sure width is divisible by 8
                new_width = (new_width // 8) * 8
        else:
            new_width = generation_width
            new_height = generation_height
        
        # Resize the image
        resized_images = []  # This was missing in the original code
        for i in range(batch_size):
            # Convert to numpy, resize, and convert back to torch
            img = image[i].cpu().numpy()
            
            # Use interpolation method based on whether we're upscaling or downscaling
            if new_width > original_width or new_height > original_height:
                # Upscaling - use bilinear interpolation
                resized = self.bilinear_resize(img, new_height, new_width)
            else:
                # Downscaling - use area interpolation
                resized = self.area_resize(img, new_height, new_width)
            
            resized_images.append(torch.from_numpy(resized))
        
        # Stack back into a batch
        result = torch.stack(resized_images)
        
        return (result, new_width, new_height)
    
    def bilinear_resize(self, img, height, width):
        """Simple bilinear resize implementation"""
        # This is a simplified version - in a real implementation you'd likely use
        # PIL, OpenCV, or torch.nn.functional.interpolate
        import cv2
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    def area_resize(self, img, height, width):
        """Area interpolation for downscaling"""
        import cv2
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

NODE_CLASS_MAPPINGS = {
    "WanVideoAutoImgResize": WanVideoAutoImgResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAutoImgResize": "WanVideo Auto Image Resize"
}