from datasets import load_dataset

# Carica il dataset
ds = load_dataset("pawkanarek/spraix_1024")

# Stampa le prime voci
print(ds)
print(ds["train"][0])
