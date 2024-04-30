import os
import numpy as np
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Dataset
import glob
from PIL import Image
import yaml
from ImageEncoder.latentdiffusion.ldm.models.autoencoder import VQModel
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def get_device():
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device

def get_model(yaml_path, ckpt_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)["model"]["params"]

    ddconfig = config['ddconfig']
    lossconfig = config['lossconfig']

    n_embed = config["n_embed"]
    embed_dim = config["embed_dim"]
    device = get_device()
    ae_model = VQModel(ddconfig, lossconfig, n_embed, embed_dim, ckpt_path=ckpt_path).to(device)

    ae_model.init_from_ckpt(ckpt_path)
    return ae_model

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = glob.glob(os.path.join(img_dir, '*'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

def encode_images(model, source_folder, dest_folder, batch_size=32):
    os.makedirs(dest_folder, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = CustomImageDataset(source_folder, transform=transform)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, (images, paths) in enumerate(tqdm(data_loader, ncols=80)):
        images = images.to(model.device)
        quant, _, _ = model.encode(images)

        for j, vec in enumerate(quant):
            filename = os.path.basename(paths[j])
            filename = os.path.splitext(filename)[0]
            filename = os.path.join(dest_folder, f'{filename}.npy')
            np.save(filename, vec.detach().cpu().numpy())

def reconstruct_images(model, encoded_folder, reconstructed_folder):
    os.makedirs(reconstructed_folder, exist_ok=True)
    encoded_files = glob.glob(os.path.join(encoded_folder, '*.npy'))
    for npy_file in tqdm(encoded_files, ncols=80):
        encoded_vector = np.load(npy_file)
        encoded_tensor = torch.from_numpy(encoded_vector)
        encoded_tensor = encoded_tensor.unsqueeze(0)
        encoded_tensor = encoded_tensor.to(model.device)
        decoded_image = model.decode(encoded_tensor)
        decoded_image = decoded_image.squeeze(0).detach().cpu().numpy()
        decoded_image = decoded_image.transpose((1, 2, 0))
        decoded_image = np.clip(decoded_image, 0, 1)
        decoded_image = Image.fromarray((decoded_image * 255).astype(np.uint8))
        filename = os.path.basename(npy_file)
        filename = os.path.splitext(filename)[0] + '.png'
        filename = os.path.join(reconstructed_folder, filename)
        decoded_image.save(filename)