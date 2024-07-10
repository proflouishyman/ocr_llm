import requests

API_KEY_FILE = "/data/lhyman6/api_key.txt"
print("Reading API key from file...")
with open(API_KEY_FILE, "r") as file:
    API_KEY = file.read().strip()
print("API key successfully read.")

# Assuming you have a function to get your API key
api_key = API_KEY

# Set up the headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Function to download the error file
def download_error_file(error_file_id):
    url = f"https://api.openai.com/v1/files/{error_file_id}/content"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        with open("error_file.json", "wb") as file:
            file.write(response.content)
        print("Error file downloaded successfully.")
    else:
        print(f"Failed to download error file. Status code: {response.status_code}")
        print(response.text)

# Download the error file
download_error_file("file-T7eEp8bTBMCSzFd4whXz0DMk")
