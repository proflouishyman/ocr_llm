import pandas as pd
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

# Using ocrenv
# modified for second download
# Define paths as variables
csv_path = '/data/lhyman6/OCR/data/gompers_corrections/by-the-people-afl-campaign-20240506 - 2of2.csv'
target_dir = '../data/second_images/'

def download_image(url, asset, target_dir):
    """
    Download a single image from the given URL and save it to the specified directory using the Asset value as the file name.
    Implements exponential backoff for retries upon failure.
    """
    filename = f"{asset}.jpg"  # Assuming JPEG format; modify as needed
    download_path = os.path.join(target_dir, filename)

    if os.path.exists(download_path):
        return filename, 'Exists'

    attempts = 0
    backoff = 1  # Start with 1 second of backoff
    max_attempts = 5  # Maximum number of retry attempts

    while attempts < max_attempts:
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(download_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return filename, 'Downloaded'
            else:
                attempts += 1
                sleep(backoff)  # Wait before retrying
                backoff *= 2  # Double the backoff interval
        except Exception as e:
            attempts += 1
            sleep(backoff)  # Wait before retrying
            backoff *= 2  # Double the backoff interval

    return filename, 'Failed after max retries'

def download_images_concurrently(max_workers=5):
    """
    Download images from a CSV file containing URLs concurrently.
    Uses exponential backoff for handling retries.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Read URLs and Assets from CSV
    data = pd.read_csv(csv_path)
    
    # Setup progress bar
    progress = tqdm(total=len(data), desc='Downloading images')
    
    # Use ThreadPoolExecutor to download images concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the download function to the URLs along with Asset
        future_to_url = {executor.submit(download_image, row['DownloadUrl'], row['Asset'], target_dir): row for index, row in data.iterrows()}
        
        # Process results as they become available
        for future in as_completed(future_to_url):
            row = future_to_url[future]
            filename, status = future.result()
            progress.update(1)
            progress.set_postfix({'file': filename, 'status': status})

    progress.close()

# Invoke the function directly from the script
download_images_concurrently()
