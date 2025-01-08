import torch
from PIL import Image
import requests
from openai import OpenAI
from transformers import (Owlv2Processor, Owlv2ForObjectDetection,
                          AutoProcessor, AutoModelForMaskGeneration)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64
import io
import numpy as np
import gradio as gr
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def encode_image_to_base64(image):
    print(f"Encode image type: {type(image)}")  # Debug print
    
    try:
        # If image is a tuple (as sometimes provided by Gradio), take the first element
        if isinstance(image, tuple):
            print(f"Image is tuple with length: {len(image)}")  # Debug print
            if len(image) > 0 and image[0] is not None:
                if isinstance(image[0], np.ndarray):
                    image = Image.fromarray(image[0])
                else:
                    image = image[0]
            else:
                raise ValueError("Invalid image tuple provided")

        # If image is a numpy array, convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # If image is a path string, open it
        elif isinstance(image, str):
            image = Image.open(image)

        print(f"Image type after conversion: {type(image)}")  # Debug print

        # Ensure image is in PIL Image format
        if not isinstance(image, Image.Image):
            raise ValueError(f"Input must be a PIL Image, numpy array, or valid image path. Got {type(image)}")

        # Convert image to RGB if it's in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Encode error details: {str(e)}")  # Debug print
        raise
        

def analyze_image(image):
    client = OpenAI(api_key=OPENAI_API_KEY)
    base64_image = encode_image_to_base64(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Your task is to determine if the image is surprising or not surprising.    
                    if the image is surprising, determine which element, figure or object in the image is making the image surprising and write it only in one sentence with no more then 6 words, otherwise, write 'NA'.    
                    Also rate how surprising the image is on a scale of 1-5, where 1 is not surprising at all and 5 is highly surprising.
                    Additionally, write one sentence about what would be expected in this scene, and one sentence about what is unexpected.
                    Provide the response as a JSON with the following structure:    
                    {
                        "label": "[surprising OR not surprising]",
                        "element": "[element]",
                        "rating": [1-5],
                        "expected": "[one sentence about what would be expected]",
                        "unexpected": "[one sentence about what is unexpected]"
                    }"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1,
        response_format={
            "type": "json_object"
        }
    )

    return response.choices[0].message.content


def show_mask(mask, ax, random_color=False):
    try:
        # Debug print to understand mask type
        print(f"show_mask input type: {type(mask)}")
        
        # Convert mask if it's a tuple
        if isinstance(mask, tuple):
            if len(mask) > 0 and mask[0] is not None:
                mask = mask[0]
            else:
                raise ValueError("Invalid mask tuple")

        # Convert torch tensor to numpy if needed
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
            
        # Handle 4D tensor/array case
        if len(mask.shape) == 4:
            mask = mask[0, 0]
        # Handle 3D tensor/array case
        elif len(mask.shape) == 3:
            mask = mask[0]

        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([1.0, 0.0, 0.0, 0.5])

        mask_image = np.zeros((*mask.shape, 4), dtype=np.float32)
        mask_image[mask > 0] = color

        ax.imshow(mask_image)
        
    except Exception as e:
        print(f"show_mask error: {str(e)}")
        print(f"mask shape: {getattr(mask, 'shape', 'no shape')}")
        raise


def process_image_detection(image, target_label, surprise_rating):
   try:
       # Handle different image input types
       if isinstance(image, tuple):
           if len(image) > 0 and image[0] is not None:
               if isinstance(image[0], np.ndarray):
                   image = Image.fromarray(image[0])
               else:
                   image = image[0]
           else:
               raise ValueError("Invalid image tuple provided")
       elif isinstance(image, np.ndarray):
           image = Image.fromarray(image)
       elif isinstance(image, str):
           image = Image.open(image)

       # Ensure image is in PIL Image format
       if not isinstance(image, Image.Image):
           raise ValueError(f"Input must be a PIL Image, got {type(image)}")

       # Ensure image is in RGB mode
       if image.mode != 'RGB':
           image = image.convert('RGB')

       device = "cuda" if torch.cuda.is_available() else "cpu"
       print(f"Using device: {device}")

       # Get original image DPI and size
       original_dpi = image.info.get('dpi', (72, 72))
       original_size = image.size
       print(f"Image size: {original_size}")

       # Calculate relative font size
       base_fontsize = min(original_size) / 80

       print("Loading models...")
       owlv2_processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14")
       owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14").to(device)
       sam_processor = AutoProcessor.from_pretrained("facebook/sam-vit-large")
       sam_model = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-large").to(device)

       print("Running object detection...")
       inputs = owlv2_processor(text=[target_label], images=image, return_tensors="pt").to(device)
       with torch.no_grad():
           outputs = owlv2_model(**inputs)

       target_sizes = torch.tensor([image.size[::-1]]).to(device)
       results = owlv2_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

       # Use original image dimensions for figure size
       dpi = 300
       width, height = image.size
       figsize = (width / dpi, height / dpi)
       
       fig = plt.figure(figsize=figsize, dpi=dpi)
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       fig.add_axes(ax)
       ax.imshow(image)

       scores = results["scores"]
       if len(scores) > 0:
           max_score_idx = scores.argmax().item()
           max_score = scores[max_score_idx].item()

           if max_score > 0.2:
               print("Processing detection results...")
               box = results["boxes"][max_score_idx].cpu().numpy()

               print("Running SAM model...")
               # Convert image to numpy array if needed for SAM
               if isinstance(image, Image.Image):
                   image_np = np.array(image)
               else:
                   image_np = image

               sam_inputs = sam_processor(
                   image_np,
                   input_boxes=[[[box[0], box[1], box[2], box[3]]]],
                   return_tensors="pt"
               ).to(device)

               with torch.no_grad():
                   sam_outputs = sam_model(**sam_inputs)

               masks = sam_processor.image_processor.post_process_masks(
                   sam_outputs.pred_masks.cpu(),
                   sam_inputs["original_sizes"].cpu(),
                   sam_inputs["reshaped_input_sizes"].cpu()
               )

               print(f"Mask type: {type(masks)}, Mask shape: {len(masks)}")
               mask = masks[0]
               if isinstance(mask, torch.Tensor):
                   mask = mask.numpy()
               
               show_mask(mask, ax=ax)

               rect = patches.Rectangle(
                   (box[0], box[1]),
                   box[2] - box[0],
                   box[3] - box[1],
                   linewidth=max(2, min(original_size) / 500),
                   edgecolor='red',
                   facecolor='none'
               )
               ax.add_patch(rect)

               # Only add the probability score
               #plt.text(
               #    box[0], box[1] - base_fontsize,
               #    f'{max_score:.2f}',
               #    color='red',
               #    fontsize=base_fontsize,
               #    fontweight='bold'
               #)

       plt.axis('off')

       print("Saving final image...")
       try:
           buf = io.BytesIO()
           fig.savefig(buf, 
                     format='png',
                     dpi=dpi,
                     bbox_inches='tight',
                     pad_inches=0)
           buf.seek(0)
           
           # Open as PIL Image
           output_image = Image.open(buf)
           
           # Convert to RGB if needed
           if output_image.mode != 'RGB':
               output_image = output_image.convert('RGB')
           
           # Save to final buffer
           final_buf = io.BytesIO()
           output_image.save(final_buf, format='PNG', dpi=original_dpi)
           final_buf.seek(0)
           
           plt.close(fig)
           buf.close()
           
           return final_buf

       except Exception as e:
           print(f"Save error details: {str(e)}")
           raise

   except Exception as e:
       print(f"Process image detection error: {str(e)}")
       print(f"Error occurred at line {e.__traceback__.tb_lineno}")
       raise

def process_and_analyze(image):
    if image is None:
        return None, "Please upload an image first."

    print(f"Initial image type: {type(image)}")  # Debug print

    if OPENAI_API_KEY is None:
        return None, "OpenAI API key not found in environment variables."

    try:
        # Convert the image to PIL format if needed
        if isinstance(image, tuple):
            print(f"Image is tuple, length: {len(image)}")  # Debug print
            if len(image) > 0 and image[0] is not None:
                if isinstance(image[0], np.ndarray):
                    image = Image.fromarray(image[0])
                else:
                    print(f"First element type: {type(image[0])}")  # Debug print
                    image = image[0]
            else:
                return None, "Invalid image format provided"
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
        
        print(f"Image type after conversion: {type(image)}")  # Debug print

        if not isinstance(image, Image.Image):
            return None, f"Invalid image format: {type(image)}"

        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Analyze image
        print("Starting GPT analysis...")  # Debug print
        gpt_response = analyze_image(image)
        print(f"GPT response: {gpt_response}")  # Debug print
        
        try:
            response_data = json.loads(gpt_response)
        except json.JSONDecodeError:
            return None, "Error: Invalid response format from GPT"

        if not all(key in response_data for key in ["label", "element", "rating"]):
            return None, "Error: Missing required fields in analysis response"

        print(f"Response data: {response_data}")  # Debug print

        if response_data["label"].lower() == "surprising" and response_data["element"].lower() != "na":
            try:
                print("Starting image detection...")  # Debug print
                result_buf = process_image_detection(image, response_data["element"], response_data["rating"])
                result_image = Image.open(result_buf)
                analysis_text = (
                    f"Label: {response_data['label']}\n"
                    f"Element: {response_data['element']}\n"
                    f"Rating: {response_data['rating']}/5\n"
                    f"Expected: {response_data['expected']}\n"
                    f"Unexpected: {response_data['unexpected']}"
            )
                return result_image, analysis_text
            except Exception as detection_error:
                print(f"Detection error details: {str(detection_error)}")  # Debug print
                return None, f"Error in image detection processing: {str(detection_error)}"
        else:
            return image, "Not Surprising"

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        detailed_error = f"Error ({error_type}): {error_msg}"
        print(detailed_error)  # Debug print
        return None, f"Error processing image: {error_msg}"

# Create Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        with gr.Row():  # Horizontal layout for the left side alignment
            with gr.Column(scale=1):  # Adjust the scale for the left section
                gr.Image(
                    value="appendix/icon.webp", 
                    width=65, 
                    interactive=False, 
                    show_label=False,
                    show_download_button=False,
                    elem_id="icon"
                )
                with gr.Column(scale=3):
                    gr.Markdown("## Image Anomaly-Surprise Detection")

        gr.Markdown(
            "This project offers a tool that identifies surprising elements in images, "
            "pinpointing what violates our expectations. It analyzes images for unexpected objects, "
            "locations, social scenarios, settings, and roles."
        )
 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image")
                analyze_btn = gr.Button("Analyze Image")

            with gr.Column():
                output_image = gr.Image(label="Processed Image")
                output_text = gr.Textbox(label="Analysis Results")

        analyze_btn.click(
            fn=process_and_analyze,
            inputs=[input_image],
            outputs=[output_image, output_text]
        )

           
        # Display example images in a row using Gradio Image components
        with gr.Row():
            gr.Image(value="appendix/gradio_example.png", width=250, show_label=False, interactive=False, show_download_button=False)
            gr.Image(value="appendix/gradio_example2.png", width=250, show_label=False, interactive=False, show_download_button=False)
            gr.Image(value="appendix/gradio_example3.png", width=250, show_label=False, interactive=False, show_download_button=False)
            gr.Image(value="appendix/gradio_example4.png", width=250, show_label=False, interactive=False, show_download_button=False)

    
    return demo





if __name__ == "__main__":
    demo = create_interface()
    demo.launch()