from datasets import load_dataset
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
import os 

if __name__ == "__main__":
    load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    dataset = load_dataset('alecocc/mathematic_games_dataset_en_2024_def', split="train")
    dataset_img = dataset.filter(lambda example: example['image'] != None)

    for item in dataset_img:
        png_image = item['image']
        png_image_id = item['id']
        # Convert to RGB mode if the image has transparency (PNG often does)
        if png_image.mode in ("RGBA", "P"):  # Check if the image has an alpha channel
            png_image = png_image.convert("RGB")

        # Save as JPEG
        os.makedirs("jpg_images", exist_ok=True)
        png_image.save(f"jpg_images/image_{png_image_id}.jpg", format="JPEG", quality=100)  # Adjust quality as needed
        # Path to your image