# Number of images to visualized
num_images=1000

# Path to the trained model checkpoint
model_checkpoint="model_19.pt"

# Path to the dataset containing images for visualization
data_folder="./data/val_images/Val_EncodedImages"

# Path to the text file corresponding to the dataset
text_file="data/val_merged_output_dict.pkl"

# Path to the folder where results will be saved
results_folder_base="./val_results"

# # Log in to wandb (if not already logged in)
# wandb login

# Loop over the number of images
for ((i = 0; i < $num_images; i++)); do
    # Run the visualization for each image
    python main.py \
    --load_path model_19.pt \
    --save_folder "$results_folder_base/val_sample_ddpm_$i" \
    --time_steps 1000 \
    --train_steps 1000 \
    --image_size 64 \
    --batch_size 1 \
    --learning_rate 1e-3 \
    --unet_dim 16 \
    --visualize
done

