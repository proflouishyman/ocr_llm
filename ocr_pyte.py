import os
from PIL import Image
import pytesseract
from tqdm import tqdm
#works

# Set the path to the Tesseract command and data directory
pytesseract.pytesseract.tesseract_cmd = r'/data/lhyman6/programs/tesseract/tesseract/bin/tesseract'
os.environ['TESSDATA_PREFIX'] = '/data/lhyman6/programs/tesseract/tesseract/tessdata'

def perform_ocr(image_path, export_directory):
    """Function to perform OCR on a single image and save the output to the specified directory."""
    base_name = os.path.basename(image_path)
    txt_path = os.path.join(export_directory, base_name.replace('.jpg', '.txt.pyte'))

    # Check if the output file already exists
    if os.path.exists(txt_path):
        return f"Skipped {image_path}: Output file already exists."

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng', config='--psm 3')
        with open(txt_path, 'w') as file:  # Always write/overwrite the txt file
            file.write(text)
        return f'OCR completed and saved for: {image_path}'
    except Exception as e:
        return f"Failed to process {image_path}: {str(e)}"

# def ensure_directory_exists(directory_path):
#     """Ensure that the directory exists and create it if it does not."""
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path, exist_ok=True)
#         print(f"Directory created: {directory_path}")

def main(task_id, directory, export_directory):
    # ensure_directory_exists(export_directory)
    images = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')])

    num_images = len(images)
    print(num_images)
    num_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT', '1'))
    images_per_task = max(1, num_images // num_tasks)
    print(images_per_task)
    start_index = task_id * images_per_task
    end_index = min(start_index + images_per_task, num_images)

    task_images = images[start_index:end_index]
    print("Start Index")
    print(start_index)
    print("End Index")
    print(end_index)

    if not task_images:
        print("No images found to process for this task.")
        return

    for index, image in enumerate(tqdm(task_images, desc="Processing images"), start=1):
        result = perform_ocr(image, export_directory)
        print(f"Image {index} of {len(task_images)}: {result}")

if __name__ == '__main__':
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))  # Default to 0 if not run by Slurm
    directory = '/data/lhyman6/data/images'
    export_directory = '/data/lhyman6/data/images'
    main(task_id, directory, export_directory)
