import os
from PIL import Image
from tqdm import tqdm

PAIRED_DIRS = {
    "low_energy_cesm": "data/paired/low_energy_cesm",
    "subtracted_cesm": "data/paired/subtracted_cesm"
}

OUTPUT_DIR = "data/processed/paired"
IMAGE_SIZE = (256, 256)


def preprocess_paired():
    for name, input_path in PAIRED_DIRS.items():
        print(f"[PAIRED] Processing: {name}")
        output_path = os.path.join(OUTPUT_DIR, name)
        os.makedirs(output_path, exist_ok=True)

        if not os.path.exists(input_path):
            print(f"[WARNING] Skipping {name}: {input_path} does not exist.")
            continue

        for filename in tqdm(os.listdir(input_path)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)

            try:
                img = Image.open(input_file).convert("L")
                img = img.resize(IMAGE_SIZE)
                img.save(output_file)
            except Exception as e:
                print(f"Failed to process {input_file}: {e}")


if __name__ == "__main__":
    preprocess_paired()
