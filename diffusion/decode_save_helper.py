import os
import sys
sys.path.append(os.pardir)
sys.path
from ImageEncoder.latentdiffusion.image_encoding_helper import get_model, reconstruct_images

def decode_save(latent_dir, reconstruction_dir):
    ae_model = get_model('../ImageEncoder/latentdiffusion/models/first_stage_models/vq-f4/config.yaml', 'ae_model.ckpt')
    reconstruct_images(ae_model, latent_dir, reconstruction_dir)