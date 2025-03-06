import torch

# Carica il modello
state_dict = torch.load("Animator2D-v2.pth", map_location='cpu')

# Stampa tutte le chiavi del modello
print("Chiavi del modello:")
for key in state_dict.keys():
    # Stampa anche la shape dei tensori
    print(f"{key}: {state_dict[key].shape}")