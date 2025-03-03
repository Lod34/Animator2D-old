import gradio as gr
import tempfile
import time
from PIL import Image, ImageDraw

def generate_animated_sprite(character_description, num_frames, character_action, viewing_direction):
    """
    Funzione di simulazione per generare un'immagine GIF animata.
    In un'implementazione reale, qui chiameresti il tuo modello di IA.
    
    Per ora, questa funzione genera solo un'immagine di esempio per testare l'interfaccia.
    """
    print(f"Generazione sprite con questi parametri:")
    print(f"- Descrizione: {character_description}")
    print(f"- Frames: {num_frames}")
    print(f"- Azione: {character_action}")
    print(f"- Direzione: {viewing_direction}")
    
    # Crea delle immagini di esempio per simulare un'animazione
    frames = []
    for i in range(num_frames):
        # Crea un'immagine di esempio
        img = Image.new('RGB', (100, 100), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        
        # Disegna qualcosa di diverso in ogni frame per simulare l'animazione
        x_offset = 10 + (i * 10) % 50
        d.ellipse((x_offset, 20, x_offset + 60, 80), fill=(255, 255, 0))
        d.text((10, 10), f"{character_action} - {i+1}", fill=(255, 255, 255))
        
        frames.append(img)
    
    # Salva i frame come GIF animata
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    frames[0].save(
        temp_file.name, 
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=200,  # 200 ms tra i frame
        loop=0  # 0 = loop infinito
    )
    temp_file.close()
    
    return temp_file.name

# Interface Gradio locale
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
        fn=generate_animated_sprite,
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
    demo.launch(share=True)  # Imposta share=False se non vuoi un URL pubblico temporaneo