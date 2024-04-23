import os
import torch
import pickle

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import random
import numpy as np

from tqdm import tqdm
# from cleanfid import fid
import wandb

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"


def cycle(dl):
    while True:
        for data in dl:
            yield data


import os
import pickle
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils import data

class CustomDataset(data.Dataset):
    def __init__(self, image_folder, text_file, image_size, data_class='', augment=False, exts=["npy"]):
        super().__init__()
        self.image_folder = image_folder
        self.text_file = text_file
        self.image_size = image_size
        
        print("image folder", self.image_folder)
        print("text file", self.text_file)
        

        # Filter paths to include only .npy files
        #self.paths = [p for p in Path(image_folder).glob("*.npy")]
        # self.paths = [p for ext in exts for p in Path(image_folder).glob(f"*.{ext}")]

        #self.paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        
        self.paths = []
        for root, dirs, files in os.walk(self.image_folder):
            for file in files:
                p = os.path.join(root, file)
                self.paths.append(p)
                
        

        with open(text_file, "rb") as f:
            self.texts = pickle.load(f)

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        
        # Load the pixel data from the .npy file
        pixel_data = np.load(img_path)

        # Create a PIL image from the pixel data
        # img = Image.fromarray(pixel_data.astype('uint8')).convert("RGB")
        # img = Image.fromarray(np.load(img_path)).convert("RGB")  # Load image from .npy file
        # img = Image.fromarray(pixel_data)
        img = torch.from_numpy(pixel_data)
        # img = self.transform(img)

        # Get corresponding text data
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        text_dict = self.texts[img_name]  # Get the dictionary associated with img_name
        text = text_dict["text"]  # Get the value associated with the key "text"

        return img, text



class LLM:
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")

    def encode_batch_1(self, texts):
        """
        Encodes a batch of texts using the BertTokenizer.

        Args:
            texts (list of str): List of input texts to encode.

        Returns:
            torch.Tensor: Tensor of token ids for the encoded texts.
        """
        print(texts)
        text_encoding = []
        for text in texts:
            text_tokens = self.tokenizer(
                text, return_tensors="pt"
            ).input_ids
            print(text_tokens.shape)
            # print(f"text_tokens.shape = {text_tokens.shape}")
            encoder_hidden_states = self.text_encoder(
                    text_tokens, return_dict=False)[0]
            # print(f"encoder_hidden_states.shape = {encoder_hidden_states.shape}")
            text_encoding.append(encoder_hidden_states)
        ret = torch.cat(text_encoding, dim=1)
        # print(f"ret.shape = {ret.shape}")
        return ret

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        text_file,
        *,
        image_size=128,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=100000,
        gradient_accumulate_every=64,
        save_and_sample_every=2000,
        results_folder="./results",
        load_path=None,
        dataset=None,
        shuffle=True,
        data_class=None,
        device=None,
        fid=False,
        val_folder=None,
        val_text_file=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.data_class = data_class

        self.fid = fid

        augment_data = False if dataset == "train" else False
        self.train_folder = folder  # image folder
        self.text_file = text_file
        self.ds = CustomDataset(folder, text_file, image_size, augment=augment_data)
        self.ds_val = CustomDataset(val_folder, val_text_file, image_size, augment=augment_data)
        self.val_indices = list(range(len(self.ds_val)))  # Store indices of validation dataset
        random.shuffle(self.val_indices)  # Shuffle indices
        print(f"dataset length: {len(self.ds)}, dataset class: {data_class}")

        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=train_batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=0,
                drop_last=True,
                collate_fn=self.custom_collate_fn
            )
        )

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if load_path != None:
            self.load(load_path)

        self.LLM = LLM()
    
    def custom_collate_fn(self, batch):
        images = [item[0] for item in batch]  # Extract image tensors
        texts = [item[1] for item in batch][0]   # Extract lists of strings
        return torch.stack(images), texts
    
    def save(self, itrs=None):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f"model.pt"))
        else:
            torch.save(data, str(self.results_folder / f"model_{itrs}.pt"))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=torch.device('cpu'))

        self.step = data["step"]
        self.model.load_state_dict(data["model"])

    def train(self):
        start_step = self.step
        self.model.train()
        self.model.to(self.device)
        """
        Training loop
        
            1. Use wandb.log to log the loss of each step 
                This loss is the average of the loss over accumulation steps 
            2. Save the sampled images every self.save_and_sample_every steps
            3. Comopute the FID score
            4. Save the model every self.save_and_sample_every steps
        """
        milestone = 0
        for self.step in tqdm(range(start_step, self.train_num_steps), desc="steps"):
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data = next(iter(self.dl))
                data_1, text = data[0].to(self.device), data[1]  # Separate image and text
                data_2 = torch.randn_like(data_1)
                
                text_encoding = self.LLM.encode_batch_1(text).to(self.device)  # Encode text

                data_1, data_2 = data_1.to(self.device), data_2.to(self.device)
                loss = torch.mean(self.model(data_1, data_2, text_encoding))
                u_loss += loss.item()
                (loss / self.gradient_accumulate_every).backward()

            # use wandb to log the loss
            wandb.log({"loss": u_loss / self.gradient_accumulate_every})

            self.opt.step()
            self.opt.zero_grad()

            if (self.step + 1) % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                save_folder = str(self.results_folder / f"sample_ddpm_{milestone}")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    
                random_text_prompts = []
                random.shuffle(self.val_indices)
                # Sample 500 text prompts
                for i in range(500):
                    # Get random index
                    index = self.val_indices[i]
                    # Get data from the dataset using the random index
                    images, texts = self.ds_val[index]
                    # Get text from the batch
                    text = texts  # Assuming batch size is 1
                    # Add text to the list
                    random_text_prompts.append(text)
                for j, text_prompt in enumerate(random_text_prompts):
                    text_encoding = self.LLM.encode_batch(text_prompt).to(self.device)
                    images = self.model.sample(1, text_encoding)
                    for _ in range(1):
                        utils.save_image(
                            images[0],
                            os.path.join(
                                save_folder, f"sample_{j}_{self.step}.png"
                            ),
                        )

                if self.fid:
                    # Make sure the images in the validation set are of the same size
                    self.val_folder = self.train_folder.replace("train", "val")
                    if len(os.listdir(self.val_folder)) == 0:
                        os.makedirs(self.val_folder, exist_ok=True)
                        # resize the images in the training set and save them in the validation set
                        for img_name in os.listdir(
                            os.path.join(self.train_folder, self.data_class)
                        ):
                            img_path = os.path.join(
                                self.train_folder, self.data_class, img_name
                            )
                            img = Image.open(img_path)
                            img = img.resize((self.image_size, self.image_size))
                            img.save(os.path.join(self.val_folder, img_name))

                    fid_score = fid.compute_fid(save_folder, self.val_folder, num_workers=0)
                    wandb.log({"fid": fid_score})
        self.save()

        images = self.model.sample(self.batch_size)
        grid = utils.make_grid(images)
        utils.save_image(
            grid, os.path.join(self.results_folder, f"sample_{milestone}.png"), nrow=6
        )
        print("training completed")

    def visualize_diffusion(
        self,
    ):
        """
        Use trained model to visualize forward and backward diffusion process

        1. Sample the first batch of images from the dataloader
        2. Visualize the forward diffusion process at 0%, 25%, 50%, 75%, 99% of the total timesteps
        3. Use the last image from the forward diffusion process to visualize the backward diffusion
            process at 0%, 25%, 50%, 75%, 99% of the total timesteps
            you can use (percent * self.model.num_timesteps) to get the timesteps
        4. Save the images in wandb
        """
        self.model.eval()
        self.model.to(self.device)

        data_1 = next(self.dl)
        data_2 = torch.randn_like(data_1)

        data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

        percent = [0.0, 0.25, 0.50, 0.75, 0.99]
        t_visualize = [int(i * self.model.num_timesteps) for i in percent]

        # visualize forward diffusion process
        img_forward = []
        for t in t_visualize:
            img = self.model.q_sample(
                data_1,
                torch.full(
                    (data_1.shape[0],), t, device=data_1.device, dtype=torch.long
                ),
                data_2,
            )
            img = torch.clamp(img, -1, 1)
            img = (img + 1) / 2
            img_forward.append(img)

        # visualize backward diffusion process
        img_backward = []
        for t in range(self.model.num_timesteps - 1, -1, -1):
            img = self.model.p_sample(
                img,
                torch.full((img.shape[0],), t, device=img.device, dtype=torch.long),
                t,
            )
            if t in t_visualize:
                img_vis = torch.clamp(img, -1, 1)
                img_vis = (img_vis + 1) / 2
                img_backward.append(img_vis)

        # save the images in wandb
        wandb.log({"forward_diffusion": [wandb.Image(img) for img in img_forward]})
        wandb.log({"backward_diffusion": [wandb.Image(img) for img in img_backward]})
        # ####################################################################
