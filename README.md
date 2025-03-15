# ComfyUI-WanVideoAutoResize

A ComfyUI custom node for automatically resizing images while maintaining aspect ratio, designed to work with WanVideo workflows.

## Installation

1. Clone this repo into `custom_nodes` folder:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/karthikg-09/ComfyUI-WanVideoAutoResize
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The node provides the following features:
- Automatically resize images while maintaining aspect ratio
- Specify target width and height
- Toggle aspect ratio preservation

### Parameters

- `image`: Input image to be resized
- `generation_width`: Target width for the output image
- `generation_height`: Target height for the output image  
- `adjust_resolution`: When enabled, maintains aspect ratio while fitting within target dimensions

### Outputs

- `image`: The resized image
- `width`: The actual width of the output image
- `height`: The actual height of the output image