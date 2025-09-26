import torch
import math
import numpy as np

class WanVideoAutoImgResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
                "preset_resolution": (["Custom", "480p", "576p", "720p", "1080p"], {
                    "default": "720p", 
                    "tooltip": "Industry standard preset resolutions"
                }),
                "resize_mode": (["Keep Proportion", "Pad", "Crop", "Stretch"], {
                    "default": "Keep Proportion",
                    "tooltip": "How to handle aspect ratio differences"
                }),
                "orientation_priority": (["Auto Detect", "Force Portrait", "Force Landscape", "Force Square"], {
                    "default": "Auto Detect",
                    "tooltip": "Smart orientation handling"
                }),
                "pad_color": (["Black", "White", "Gray", "Blur"], {
                    "default": "Black",
                    "tooltip": "Padding color when using Pad mode"
                }),
                "interpolation": (["Bilinear", "Bicubic", "Lanczos", "Nearest"], {
                    "default": "Bilinear",
                    "tooltip": "Interpolation method: Bilinear (fast), Bicubic (quality), Nearest (fastest), Lanczos (high quality)"
                })
            },
            "optional": {
                "custom_width": ("INT", {
                    "default": 832, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Custom width (only used when preset is 'Custom')"
                }),
                "custom_height": ("INT", {
                    "default": 480, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Custom height (only used when preset is 'Custom')"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "orientation", "aspect_ratio", "resize_info")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    OUTPUT_NODE = True

    def get_preset_dimensions(self, preset, orientation):
        """Get dimensions for preset resolutions based on orientation"""
        presets = {
            "480p": (854, 480),    # SD
            "576p": (1024, 576),   # PAL
            "720p": (1280, 720),   # HD
            "1080p": (1920, 1080)  # Full HD
        }
        
        if preset not in presets:
            return None
            
        width, height = presets[preset]
        
        # Adjust for orientation
        if orientation == "portrait":
            return (height, width)  # Swap dimensions for portrait
        elif orientation == "square":
            # For square, use the smaller dimension
            size = min(width, height)
            return (size, size)
        else:  # landscape
            return (width, height)
    
    def detect_orientation(self, width, height):
        """Detect image orientation with improved range detection"""
        aspect_ratio = width / height
        
        # More nuanced detection with better ranges
        # For example: 410x390 has aspect ratio ~1.05, should be considered square
        if 0.85 <= aspect_ratio <= 1.18:  # Expanded square range
            return "square"
        elif aspect_ratio > 1.18:  # Clear landscape
            return "landscape"
        else:  # aspect_ratio < 0.85, clear portrait
            return "portrait"
    
    def apply_orientation_priority(self, detected_orientation, priority, original_width, original_height):
        """Apply orientation priority logic"""
        if priority == "Auto Detect":
            return detected_orientation
        elif priority == "Force Portrait":
            return "portrait"
        elif priority == "Force Landscape":
            return "landscape"
        elif priority == "Force Square":
            return "square"
        return detected_orientation
    
    def calculate_resize_dimensions(self, original_width, original_height, target_width, target_height, resize_mode):
        """Calculate final dimensions based on resize mode"""
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height
        
        if resize_mode == "Stretch":
            return target_width, target_height, "stretched"
        elif resize_mode == "Keep Proportion":
            if original_aspect > target_aspect:
                # Fit to width
                new_width = target_width
                new_height = int(target_width / original_aspect)
            else:
                # Fit to height
                new_height = target_height
                new_width = int(target_height * original_aspect)
            # Ensure divisible by 8
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            return new_width, new_height, "proportional"
        elif resize_mode == "Crop":
            if original_aspect > target_aspect:
                # Crop width
                new_height = target_height
                new_width = int(target_height * original_aspect)
            else:
                # Crop height
                new_width = target_width
                new_height = int(target_width / original_aspect)
            return target_width, target_height, "cropped"
        elif resize_mode == "Pad":
            # Same as Keep Proportion, but we'll pad later
            if original_aspect > target_aspect:
                new_width = target_width
                new_height = int(target_width / original_aspect)
            else:
                new_height = target_height
                new_width = int(target_height * original_aspect)
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            return new_width, new_height, "padded"
        
        return target_width, target_height, "default"
    
    def resize_image_tensor(self, image, new_width, new_height, interpolation='bilinear'):
        """Fast tensor resizing with GPU optimization"""
        # Convert to [batch, channels, height, width] for maximum GPU efficiency
        image_gpu = image.permute(0, 3, 1, 2).contiguous()
        
        # Ensure GPU placement for speed
        if torch.cuda.is_available() and not image_gpu.is_cuda:
            image_gpu = image_gpu.cuda()
        
        # Map interpolation methods
        mode_map = {
            'bilinear': 'bilinear',
            'bicubic': 'bicubic',
            'lanczos': 'bicubic',  # Use bicubic as closest alternative
            'nearest': 'nearest'
        }
        mode = mode_map.get(interpolation.lower(), 'bilinear')
        
        # Resize with GPU acceleration
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            if mode == 'bicubic':
                resized = torch.nn.functional.interpolate(
                    image_gpu,
                    size=(new_height, new_width),
                    mode='bicubic',
                    align_corners=True
                )
            else:
                resized = torch.nn.functional.interpolate(
                    image_gpu,
                    size=(new_height, new_width),
                    mode=mode,
                    align_corners=False
                )
        
        # Convert back to [batch, height, width, channels]
        return resized.permute(0, 2, 3, 1).contiguous()
    
    
    def apply_padding_fast(self, image, target_width, target_height, pad_color):
        """Ultra-fast padding optimized for video processing"""
        batch_size, current_height, current_width, channels = image.shape
        
        # Quick exit if no padding needed
        if current_width == target_width and current_height == target_height:
            return image
        
        # Calculate padding efficiently
        pad_width = max(0, target_width - current_width)
        pad_height = max(0, target_height - current_height)
        
        if pad_width == 0 and pad_height == 0:
            return image
        
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        
        # Convert to GPU tensor format for maximum speed
        image_gpu = image.permute(0, 3, 1, 2).contiguous()
        
        # Fast padding colors
        if pad_color == "White":
            pad_value = 1.0
        elif pad_color == "Gray":
            pad_value = 0.5
        elif pad_color == "Blur":
            # Fast reflection padding - much faster than blur
            padded = torch.nn.functional.pad(
                image_gpu, 
                (pad_left, pad_right, pad_top, pad_bottom), 
                mode='reflect'
            )
            return padded.permute(0, 2, 3, 1).contiguous()
        else:  # Black (default)
            pad_value = 0.0
        
        # Ultra-fast constant padding
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            padded = torch.nn.functional.pad(
                image_gpu, 
                (pad_left, pad_right, pad_top, pad_bottom), 
                mode='constant', 
                value=pad_value
            )
        
        return padded.permute(0, 2, 3, 1).contiguous()
    
    
    def apply_crop_fast(self, image, target_width, target_height):
        """Ultra-fast center crop optimized for video frames"""
        batch_size, current_height, current_width, channels = image.shape
        
        # Quick exit if no cropping needed
        if current_width == target_width and current_height == target_height:
            return image
        
        # Calculate crop area with bounds checking
        start_x = max(0, (current_width - target_width) // 2)
        start_y = max(0, (current_height - target_height) // 2)
        end_x = min(current_width, start_x + target_width)
        end_y = min(current_height, start_y + target_height)
        
        # Ensure we don't exceed bounds
        if end_x - start_x != target_width:
            start_x = max(0, current_width - target_width)
            end_x = start_x + target_width
        if end_y - start_y != target_height:
            start_y = max(0, current_height - target_height)
            end_y = start_y + target_height
        
        # Ultra-fast tensor slicing - much faster than copying
        return image[:, start_y:end_y, start_x:end_x, :].contiguous()

    def process(self, image, preset_resolution, resize_mode, orientation_priority, pad_color, 
                interpolation, custom_width=832, custom_height=480):
        # Handle different image dimensions with video optimization
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Ensure tensor is on GPU and contiguous for maximum speed
        if torch.cuda.is_available() and not image.is_cuda:
            image = image.cuda()
        image = image.contiguous()
        
        batch_size, original_height, original_width, channels = image.shape
        original_aspect = original_width / original_height
        
        
        # Detect original orientation
        detected_orientation = self.detect_orientation(original_width, original_height)
        
        # Apply orientation priority
        final_orientation = self.apply_orientation_priority(
            detected_orientation, orientation_priority, original_width, original_height
        )
        
        # Get target dimensions
        if preset_resolution == "Custom":
            target_width, target_height = custom_width, custom_height
        else:
            dimensions = self.get_preset_dimensions(preset_resolution, final_orientation)
            if dimensions is None:
                target_width, target_height = custom_width, custom_height
            else:
                target_width, target_height = dimensions
        
        # Calculate resize dimensions based on mode
        resize_width, resize_height, resize_info = self.calculate_resize_dimensions(
            original_width, original_height, target_width, target_height, resize_mode
        )
        
        # Fast processing pipeline
        
        if resize_mode == "Stretch":
            result = self.resize_image_tensor(image, target_width, target_height, interpolation)
            final_width, final_height = target_width, target_height
            
        elif resize_mode == "Keep Proportion":
            result = self.resize_image_tensor(image, resize_width, resize_height, interpolation)
            final_width, final_height = resize_width, resize_height
            
        elif resize_mode == "Pad":
            resized = self.resize_image_tensor(image, resize_width, resize_height, interpolation)
            result = self.apply_padding_fast(resized, target_width, target_height, pad_color)
            final_width, final_height = target_width, target_height
            
        elif resize_mode == "Crop":
            if resize_width == target_width and resize_height == target_height:
                result = self.resize_image_tensor(image, target_width, target_height, interpolation)
            else:
                # Scale up then crop
                scale_factor = max(target_width / original_width, target_height / original_height)
                temp_width = max(target_width, int(original_width * scale_factor))
                temp_height = max(target_height, int(original_height * scale_factor))
                
                # Ensure dimensions are efficient for GPU processing
                temp_width = ((temp_width + 7) // 8) * 8
                temp_height = ((temp_height + 7) // 8) * 8
                
                resized = self.resize_image_tensor(image, temp_width, temp_height, interpolation)
                result = self.apply_crop_fast(resized, target_width, target_height)
            
            final_width, final_height = target_width, target_height
        
        
        # Create detailed info string
        # Processing info
        info_parts = [
            f"Original: {original_width}x{original_height}",
            f"Target: {target_width}x{target_height}",
            f"Final: {final_width}x{final_height}",
            f"Mode: {resize_mode}",
            f"Orientation: {detected_orientation}→{final_orientation}",
            f"Interpolation: {interpolation}"
        ]
        
        if resize_mode == "Pad":
            info_parts.append(f"Pad: {pad_color}")
        
        resize_info_str = " | ".join(info_parts)
        
        # Ensure result is properly formatted and contiguous
        result = result.contiguous()
        
        return (
            result, 
            final_width, 
            final_height, 
            final_orientation,
            round(final_width / final_height, 3),
            resize_info_str
        )

NODE_CLASS_MAPPINGS = {
    "WanVideoAutoImgResize": WanVideoAutoImgResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAutoImgResize": "WanVideo Auto Image Resize"
}