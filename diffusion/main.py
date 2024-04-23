# from comet_ml import Experiment
from trainer import Trainer
from diffusion import Diffusion
# from unet import Unet
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
import argparse
import wandb
import torch

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
# fix random seed for reproducibility
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_steps",
        default=1000,
        type=int,
        help="The number of steps the scheduler takes to go from clean image to an isotropic gaussian. This is also the number of steps of diffusion.",
    )
    parser.add_argument(
        "--train_steps",
        default=50000,
        type=int,
        help="The number of iterations for training.",
    )
    parser.add_argument("--save_folder", default="./results_afhq", type=str)
    parser.add_argument("--data_path", default="data/train_images/Train_EncodedImages/", type=str)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument(
        "--data_class", choices=["all", "cat", "dog", "wild"], default="", type=str
    )
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--text_path", default='data/training_merged_output_dict.pkl', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--unet_dim", default=64, type=int)
    parser.add_argument("--unet_dim_mults", nargs="+", default=[1, 2, 4, 8], type=int)
    parser.add_argument("--fid", action="store_true")
    parser.add_argument(
        "--save_and_sample_every",
        default=1000,
        type=int,
        help="The number of steps between periodically saving the model state, " + \
             "sampling example images, and optional calculating FID"
    )

    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    print(args)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    wandb.login()
    wandb.init(
        project="IDL_Project",
        config=args.__dict__,
        reinit=True,
        name=args.save_folder.split("/")[-1],
    )

    model = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch.float16
        ).to(device)
    diffusion = Diffusion(
        model,
        image_size=args.image_size,
        channels=3,
        timesteps=args.time_steps,  # number of steps
    ).to(device)

    trainer = Trainer(
        diffusion,
        args.data_path,
        image_size=args.image_size,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.train_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        results_folder=args.save_folder,
        load_path=args.load_path,
        dataset="train",
        data_class=args.data_class,
        device=device,
        shuffle=False,
        save_and_sample_every=args.save_and_sample_every,
        fid=args.fid,
        val_folder="data/val_images/Val_EncodedImages/",
        text_file="data/training_merged_output_dict.pkl",
        val_text_file="data/val_merged_output_dict.pkl",
    )

    if args.visualize:
        if args.load_path is None:
            print("No model to visualize, Please provide a load path.")
            exit(0)
        trainer.visualize_diffusion()
    else:
        trainer.train()
