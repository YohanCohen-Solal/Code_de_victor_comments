import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BasicDataset(Dataset): # De quoi est composé la classe Dataset
    def __init__(self, image_dir, mask_dir, transform=None): # Initialisation des images
        self.image_dir = image_dir # images sans mask
        self.mask_dir = mask_dir # images avec mask
        self.transform = transform
        self.images = os.listdir(image_dir) # récupère le nom de chaque image sans mask en prenant le chemin du dossier

    def __len__(self): # retourne la taille de chaque image en fonction du nombre de pixels
        return len(self.images) 

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) # Permet de joindre le chemin de deux items. Ex: récupère le chemin d'un dossier + le nom d'une image
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.jpg")) # Same en replaçant l'extension et le nom de l'image
        image = np.array(Image.open(img_path).convert("RGB")) # ouvre l'image et la converti en RGB
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #Convertir en niveau de gris
        mask[mask == 1.0] = 100.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

print(mask)