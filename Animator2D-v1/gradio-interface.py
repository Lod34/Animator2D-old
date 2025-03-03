import os
import gradio as gr
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import tempfile

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize model globally with error handling
try:
    print("Loading model...")
    model = DiffusionPipeline.from_pretrained(
        "Lod34/Animator2D",
        torch_dtype=torch.float32  # Use float32 for better compatibility
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def generate_sprite(description, action, direction, num_frames):
    """Generate sprite animation"""
    try:
        # Create prompt
        prompt = f"A sprite of {description} {action}, facing {direction}"
        
        # Generate frames
        result = model(prompt, num_frames=num_frames)
        
        # Save as GIF
        temp_path = os.path.join(tempfile.gettempdir(), "animation.gif")
        frames = [Image.fromarray(frame) for frame in result.frames]
        frames[0].save(
            temp_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        return temp_path
    except Exception as e:
        return None

def create_interface():
    """Create and launch the Gradio interface."""
    
    with gr.Blocks(title="Animator2D Sprite Generator") as interface:
        gr.Markdown("# Animator2D Sprite Generator")
        gr.Markdown("Generate animated sprites using AI")
        
        with gr.Row():
            with gr.Column():
                # Input components
                description = gr.Textbox(
                    label="Sprite Description",
                    placeholder="E.g., a cute pixel art cat"
                )
                action = gr.Textbox(
                    label="Sprite Action",
                    placeholder="E.g., walking, jumping"
                )
                direction = gr.Dropdown(
                    label="Direction",
                    choices=["North", "South", "East", "West"],
                    value="South"
                )
                num_frames = gr.Slider(
                    label="Number of Frames",
                    minimum=2,
                    maximum=24,
                    value=8,
                    step=1
                )
                generate_btn = gr.Button("Generate Animation")
            
            with gr.Column():
                # Output components
                output_image = gr.Image(label="Generated Animation", type="filepath")
        
        # Connect components
        generate_btn.click(
            fn=generate_sprite,
            inputs=[description, action, direction, num_frames],
            outputs=output_image
        )
        
    return interface

# Launch the application
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)