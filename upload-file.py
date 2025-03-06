from huggingface_hub import HfApi, login
import json
import os

MODEL_NAME = "Animator2D-v3"

def upload_model_to_hf():
    # 1. Login to Hugging Face
    print("Effettuo il login su Hugging Face...")
    login()  # Ti chiederà di inserire il token di Hugging Face

    # 2. Initialize the API
    api = HfApi()
    
    # 3. Prepare config
    print("Creo il file di configurazione...")
    config = {
        "model_type": "sprite_generator",
        "architectures": ["SpriteGenerator"],
        "text_encoder_name": "t5-base",
        "latent_dim": 512,
        "image_size": 256,
        "num_channels": 3,
        "max_frames": 8
    }
    
    # Salva config localmente
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 4. Upload files
    print("Carico i file su Hugging Face...")
    
    # Carica il modello - usando il nome esatto del file
    print("Carico il modello...")
    api.upload_file(
        path_or_fileobj=f"./{MODEL_NAME}/{MODEL_NAME}.pth",  # Nome esatto del file
        path_in_repo="pytorch_model.bin",
        repo_id=f"Lod34/{MODEL_NAME}",
        repo_type="model"
    )
    
    # Carica il config
    print("Carico il file di configurazione...")
    api.upload_file(
        path_or_fileobj="config.json",
        path_in_repo="config.json",
        repo_id=f"Lod34/{MODEL_NAME}",
        repo_type="model"
    )
    
    print("Upload completato con successo!")
    
    # 5. Cleanup
    if os.path.exists('config.json'):
        os.remove('config.json')

if __name__ == "__main__":
    upload_model_to_hf()