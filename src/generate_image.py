from diffusers import AutoPipelineForText2Image
import torch

class ImageGenerationModel:
    def __init__(self, model_name="stabilityai/sdxl-turbo", device="cuda"):
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16")
        self.pipe.to(device)

    def generate_image(self, prompt, num_inference_steps=1, guidance_scale=0.0):
        return self.pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

def generate_image_from_text(text, image_size=(800, 600), font_size=20, font_path=None):
    # Create a blank image with white background
    model = ImageGenerationModel()

    # Generate an image using the model
    image = model.generate_image(text)

    return image

# Example usage
if __name__ == "__main__":
    text_description = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    image = generate_image_from_text(text_description)
    image.show()  # Display the image
    image.save('/Users/river/Desktop/Project/DuoMod/src/output_image.png')  # Save the image