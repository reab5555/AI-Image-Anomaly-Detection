# Image Surprise Analyzer

This project offers a tool that identifies surprising elements in images, pinpointing what violates our expectations. It analyzes images for unexpected objects, locations, social scenarios, settings, and roles.

## Description

The Image Surprise Analyzer is a pipeline designed to automatically detect surprising or unexpected elements within an image. It goes beyond simple object recognition and delves into the context of the scene, identifying elements that defy expectations. The tool can be useful for various applications, such as generating creative content, identifying unusual events, or gaining a deeper understanding of visual information. This is achieved by analyzing images using a combination of large language models (LLMs) and object detection models, leveraging their capabilities to understand and interpret both the visual and conceptual information within images. The tool is built with a user-friendly interface using Gradio, allowing for easy interaction and experimentation.

## Technical Details

The system operates as a pipeline, integrating multiple models to achieve its objective. It starts by taking an image as input, which is then passed to an LLM (specifically, the `gpt-4o-mini` model). The LLM analyzes the image and determines if it contains any surprising elements, along with a rating of the surprise level (on a scale of 1 to 5) and describes what is expected and what is unexpected about the scene. If the image is deemed surprising by the LLM, it also extracts the surprising element. If so, the tool uses the identified element as a prompt to an object detection model, specifically the `owlv2-base-patch16` model, to detect the object in the image. To refine the detection further, the identified bounding box is used as a prompt for the `sam-vit-base` model to segment the surprising object, producing an accurate mask highlighting the unexpected element within the scene. The identified mask, along with the bounding box is overlayed on the original image, and returned along with the analysis text. This is done using a combination of `torch`, `PIL`, `transformers`, `OpenAI`, `matplotlib`, `base64`, `io` and `numpy`. Gradio is used to create the graphical user interface.

## Evaluation

To evaluate the performance of our different LLMs in identifying surprising images, a study was conducted. First, a set of images was created, which contained examples of surprising images and non-surprising images (300 images per class). Then, human labelers independently reviewed each image and determined whether it was surprising or not, and the degree of surprise on a likert scale from 1 to 5. This created a ground truth consensus for each image, allowing us to measure the models' performance.

The performance of different models was assessed for identifying surprising vs. non-surprising images. The following table shows some of the results with the best performing models:

| Model                                                 | Temperature | Accuracy | Surprising (Presicion) | Not-Surprising (Presicion) |
|-------------------------------------------------------|----------|----------|------------|----------------|
| `GPT-4o-mini`                            | 0.1      | 0.84     | 0.79       | 0.9            |
| `GPT-4o`                                  | 0.1      | 0.78 | 0.74       | 0.85           |
| `Llama-3.2-11B-Vision`                           | 0.1      | 0.62     | 0.83       | 0.57           |

This project is still under development, but contributions are welcomed to help improve its performance and capabilities.
