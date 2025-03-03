import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import io

# Definiamo un percorso per salvare il modello addestrato
MODEL_PATH = "sprite_generator_model"
os.makedirs(MODEL_PATH, exist_ok=True)

# Carichiamo il dataset da Hugging Face
print("Caricamento del dataset...")
dataset = load_dataset("pawkanarek/spraix_1024")
print(f"Dataset caricato. Dimensioni: {len(dataset['train'])} esempi di training")

# Verifichiamo gli split disponibili
print("Split disponibili nel dataset:")
print(dataset.keys())

# Stampiamo un esempio per capire la struttura del dataset
print("Esempio di dato dal dataset:")
example = dataset['train'][0]
print("Chiavi disponibili:", example.keys())
for key in example:
    print(f"{key}: {type(example[key])}")
    # Se il valore è un dizionario, stampiamo anche le sue chiavi
    if isinstance(example[key], dict):
        print(f"  Sottochavi: {example[key].keys()}")

# Classe per il nostro dataset personalizzato
class SpriteDataset(Dataset):
    def __init__(self, dataset_to_use, max_length=128):
        self.dataset = dataset_to_use
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float), # Converti in float32
            transforms.Lambda(lambda image: image[:3, :, :]), # Seleziona solo i primi 3 canali (RGB)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Estrai informazioni dalla descrizione completa
        description = item['text'] if 'text' in item else ""

        # Estrai numero di frame dal testo
        num_frames = 1  # valore di default
        if "frame" in description:
            # Cerca numeri seguiti da "frame" nel testo
            import re
            frames_match = re.search(r'(\d+)-frame', description)
            if frames_match:
                num_frames = int(frames_match.group(1))

        # Prepara il testo per il modello
        text_input = f"""
        Description: {description}
        Number of frames: {num_frames}
        """

        # Tokenizziamo l'input testuale
        encoded_text = self.tokenizer(
            text_input,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Prepariamo l'immagine (o le immagini se ci sono frame multipli)
        sprite_frames = []

        # Controlla le chiavi disponibili per i frame
        if 'image' in item:
            # Se c'è un'unica immagine
            img = item['image']
            if isinstance(img, dict) and 'bytes' in img:
                img_pil = Image.open(io.BytesIO(img['bytes']))
                sprite_frames.append(self.transform(img_pil))
            elif hasattr(img, 'convert'):  # Se è già un'immagine PIL
                sprite_frames.append(self.transform(img))
        else:
            # Prova a cercare frame_0, frame_1, ecc.
            for frame in range(num_frames):
                frame_key = f'frame_{frame}'
                if frame_key in item:
                    img = item[frame_key]
                    if isinstance(img, dict) and 'bytes' in img:
                        img_pil = Image.open(io.BytesIO(img['bytes']))
                        sprite_frames.append(self.transform(img_pil))
                    elif hasattr(img, 'convert'):  # Se è già un'immagine PIL
                        sprite_frames.append(self.transform(img))

        # Se non abbiamo trovato immagini, prova a cercare altre chiavi comuni
        if not sprite_frames:
            possible_image_keys = ['image', 'img', 'sprite', 'frames']
            for key in possible_image_keys:
                if key in item and item[key] is not None:
                    img = item[key]
                    if isinstance(img, dict) and 'bytes' in img:
                        img_pil = Image.open(io.BytesIO(img['bytes']))
                        sprite_frames.append(self.transform(img_pil))
                    elif hasattr(img, 'convert'):  # Se è già un'immagine PIL
                        sprite_frames.append(self.transform(img))
                    break

        # Se ancora non abbiamo frame, crea un tensore vuoto
        if not sprite_frames:
            sprite_frames.append(torch.zeros((3, 256, 256)))

        # Combiniamo tutti i frame in un unico tensore
        sprite_tensor = torch.stack(sprite_frames)

        return {
            "input_ids": encoded_text.input_ids.squeeze(),
            "attention_mask": encoded_text.attention_mask.squeeze(),
            "sprite_frames": sprite_tensor,
            "num_frames": torch.tensor(num_frames, dtype=torch.int64)
        }

# Modello generatore di sprite
class SpriteGenerator(nn.Module):
    def __init__(self, text_encoder_name="t5-base", latent_dim=512):
        super(SpriteGenerator, self).__init__()

        # Encoder testuale
        self.text_encoder = AutoModelForSeq2SeqLM.from_pretrained(text_encoder_name)
        # Freeziamo i parametri dell'encoder per iniziare
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Proiezione dal testo al latent space
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.d_model, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim)
        )

        # Frame generator (una rete deconvoluzionale)
        self.generator = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # -> 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # -> 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # -> 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # -> 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # -> 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # -> 16 x 128 x 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),  # -> 3 x 256 x 256
            nn.Tanh()
        )

        # Frame interpolator per supportare animazioni con più frame
        self.frame_interpolator = nn.Sequential(
            nn.Linear(latent_dim + 1, latent_dim),  # +1 per l'informazione sul frame
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input_ids, attention_mask, num_frames=1):
        batch_size = input_ids.shape[0]

        # Codifichiamo il testo
        text_outputs = self.text_encoder.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Utilizziamo l'ultimo hidden state
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # Media per ottenere un vettore per esempio

        # Proiettiamo nello spazio latente
        latent_vector = self.text_projection(text_features)

        # Generiamo frame multipli se necessario
        all_frames = []
        for frame_idx in range(max(num_frames.max().item(), 1)):
            # Normalizziamo l'indice del frame
            frame_info = torch.ones((batch_size, 1), device=latent_vector.device) * frame_idx / max(num_frames.max().item(), 1)

            # Combiniamo il vettore latente con l'informazione sul frame
            frame_latent = self.frame_interpolator(
                torch.cat([latent_vector, frame_info], dim=1)
            )

            # Ricordiamo quanti frame generare per ogni esempio del batch
            frame_mask = (frame_idx < num_frames).float().unsqueeze(1)

            # Riformattiamo per il generatore
            frame_latent_reshaped = frame_latent.unsqueeze(2).unsqueeze(3)  # [B, latent_dim, 1, 1]

            # Generiamo il frame
            frame = self.generator(frame_latent_reshaped) * frame_mask.unsqueeze(2).unsqueeze(3)
            all_frames.append(frame)

        # Combiniamo tutti i frame
        sprites = torch.stack(all_frames, dim=1)  # [B, num_frames, 3, 256, 256]

        return sprites

# Funzione per addestrare il modello
def train_model(model, train_loader, val_loader, epochs=10, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_sprites = batch["sprite_frames"].to(device)
            num_frames = batch["num_frames"].to(device)

            optimizer.zero_grad()

            # Forward pass
            output_sprites = model(input_ids, attention_mask, num_frames)

            # Calcoliamo la loss per il batch
            loss = 0.0
            for i in range(len(num_frames)):
                # Utilizziamo solo i frame validi per ogni esempio
                valid_frames = min(output_sprites.shape[1], target_sprites.shape[1], num_frames[i].item())
                if valid_frames > 0:
                    loss += criterion(
                        output_sprites[i, :valid_frames],
                        target_sprites[i, :valid_frames]
                    )

            loss = loss / len(num_frames)  # Media per batch

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_sprites = batch["sprite_frames"].to(device)
                num_frames = batch["num_frames"].to(device)

                output_sprites = model(input_ids, attention_mask, num_frames)

                # Calcoliamo la loss per il batch di validazione
                loss = 0.0
                for i in range(len(num_frames)):
                    valid_frames = min(output_sprites.shape[1], target_sprites.shape[1], num_frames[i].item())
                    if valid_frames > 0:
                        loss += criterion(
                            output_sprites[i, :valid_frames],
                            target_sprites[i, :valid_frames]
                        )

                loss = loss / len(num_frames)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Salviamo il modello migliore
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, "best_model.pth"))
            print(f"Modello salvato con Val Loss: {val_loss:.4f}")

    # Salviamo il modello finale
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "Animator2D-v2.pth"))
    print(f"Addestramento completato. Modello finale salvato.")

    return model

# Codice per l'esecuzione dell'addestramento
if __name__ == "__main__":
    # Dividiamo il dataset in train e validation manualmente
    # dato che abbiamo solo lo split "train"
    train_size = int(0.8 * len(dataset['train']))  # 80% per training
    val_size = len(dataset['train']) - train_size   # 20% per validation

    print(f"Dividendo il dataset: {train_size} esempi per training, {val_size} esempi per validation")

    # Creiamo i subset
    train_subset, val_subset = random_split(
        dataset['train'],
        [train_size, val_size]
    )

    # Creiamo i dataset personalizzati
    train_dataset = SpriteDataset(train_subset)
    val_dataset = SpriteDataset(val_subset)

    print(f"Dataset creati: {len(train_dataset)} esempi di training, {len(val_dataset)} esempi di validation")

    # Creiamo i dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Creiamo e addestriamo il modello
    model = SpriteGenerator()
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=20
    )

    print("Modello addestrato con successo!")