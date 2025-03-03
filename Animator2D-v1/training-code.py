import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SpriteDataset(Dataset):
    def __init__(self, dataset_split="train"):
        # Load the dataset from HuggingFace
        self.dataset = load_dataset("pawkanarek/spraix_1024", split=dataset_split)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize all sprites to same size
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Process text description
        text = f"{item['text']}"  # Contains frames, description, action, direction
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        
        # Process image
        # The item['image'] is already a PIL Image. Convert it to RGB if it's not already
        image = item['image'].convert('RGB')  
        # Removed Image.fromarray as it's unnecessary
        image_tensor = self.transform(image)
        
        return {
            'text_ids': encoded_text['input_ids'].squeeze(),
            'text_mask': encoded_text['attention_mask'].squeeze(),
            'image': image_tensor
        }

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 512)  # Reduce BERT output dimension
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.linear(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token

class SpriteGenerator(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        self.generator = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 3 x 64 x 64
        )
        
    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        return self.generator(z)

class Animator2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.sprite_generator = SpriteGenerator()
        
    def forward(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)
        generated_sprite = self.sprite_generator(text_features)
        return generated_sprite

def train_model(num_epochs=100, batch_size=32, learning_rate=0.0002):
    # Initialize dataset and dataloader
    dataset = SpriteDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Animator2D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            real_images = batch['image'].to(device)
            
            # Forward pass
            generated_images = model(text_ids, text_mask)
            
            # Calculate loss
            loss = criterion(generated_images, real_images)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "animator2d_model.pth")
    return model

def generate_sprite_animation(model, num_frames, description, action, direction):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Prepare input text
    text = f"{num_frames}-frame sprite animation of: {description}, that: {action}, facing: {direction}"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded_text = tokenizer(
        text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    
    # Generate sprite sheet
    with torch.no_grad():
        text_ids = encoded_text['input_ids'].to(device)
        text_mask = encoded_text['attention_mask'].to(device)
        generated_sprite = model(text_ids, text_mask)
        
    # Convert to image
    generated_sprite = generated_sprite.cpu().squeeze(0)
    generated_sprite = (generated_sprite + 1) / 2  # Denormalize
    generated_sprite = transforms.ToPILImage()(generated_sprite)
    
    return generated_sprite

# Example usage
if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    # Generate a new sprite animation
    test_params = {
        "num_frames": 17,
        "description": "red-haired hobbit in green cape",
        "action": "shoots with slingshot",
        "direction": "East"
    }
    
    sprite_sheet = generate_sprite_animation(
        model,
        test_params["num_frames"],
        test_params["description"],
        test_params["action"],
        test_params["direction"]
    )
    sprite_sheet.save("generated_sprite.png")