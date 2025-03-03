import gradio as gr

# Define the function to generate the sprite based on user input
def generate_sprite(character_description, num_frames, character_action, viewing_direction):
    # Combine user inputs into a single prompt
    prompt = f"Character description: {character_description}\n" \
             f"Character action: {character_action}\n" \
             f"Viewing direction: {viewing_direction}\n" \
             f"Number of frames: {num_frames}"
    
    # Load the model from Hugging Face Hub
    model = gr.Interface.load("huggingface/Lod34/Animator2D-v2")
    
    # Generate the sprite using the model
    result = model(prompt)
    
    return result

# Configure the Gradio interface
with gr.Blocks(title="Animated Sprite Generator") as demo:
    gr.Markdown("# 🎮 AI Animated Sprite Generator")
    gr.Markdown("""
    This tool uses an AI model to generate animated sprites based on text descriptions.
    Enter the character description, number of frames, character action, and viewing direction to generate your animated sprite.
    """)
    
    with gr.Row():
        with gr.Column():
            # Input components
            char_desc = gr.Textbox(label="Character Description", 
                                   placeholder="Ex: a knight with golden armor and a fire sword",
                                   lines=3)
            num_frames = gr.Slider(minimum=1, maximum=8, step=1, value=4, 
                                   label="Number of Animation Frames")
            char_action = gr.Dropdown(
                choices=["idle", "walk", "run", "attack", "jump", "die", "cast spell", "dance"],
                label="Character Action",
                value="idle"
            )
            view_direction = gr.Dropdown(
                choices=["front", "back", "left", "right", "front-left", "front-right", "back-left", "back-right"],
                label="Viewing Direction",
                value="front"
            )
            generate_btn = gr.Button("Generate Animated Sprite")
        
        with gr.Column():
            # Output component
            animated_output = gr.Image(label="Animated Sprite (GIF)")
    
    # Connect the button to the function
    generate_btn.click(
        fn=generate_sprite,
        inputs=[char_desc, num_frames, char_action, view_direction],
        outputs=animated_output
    )
    
    # Predefined examples
    gr.Examples(
        [
            ["A wizard with blue cloak and pointed hat", 4, "cast spell", "front"],
            ["A warrior with heavy armor and axe", 6, "attack", "right"],
            ["A ninja with black clothes and throwing stars", 8, "run", "front-left"],
            ["A princess with golden crown and pink dress", 4, "dance", "front"]
        ],
        inputs=[char_desc, num_frames, char_action, view_direction]
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(share=True)
