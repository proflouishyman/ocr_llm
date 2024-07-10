import os
import shutil

def delete_files_in_subdirectory(subdirectory):
    """Delete all files in the specified subdirectory."""
    if os.path.exists(subdirectory) and os.path.isdir(subdirectory):
        for filename in os.listdir(subdirectory):
            file_path = os.path.join(subdirectory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Deleted directory: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def copy_file_to_parent_directory(file_path):
    """Copy the specified file to one directory higher."""
    if os.path.exists(file_path) and os.path.isfile(file_path):
        parent_directory = os.path.dirname(os.path.dirname(file_path))
        destination_path = os.path.join(parent_directory, os.path.basename(file_path))
        try:
            shutil.copy(file_path, destination_path)
            print(f"Copied file to: {destination_path}")
        except Exception as e:
            print(f"Failed to copy {file_path} to {destination_path}. Reason: {e}")
    else:
        print(f"File does not exist: {file_path}")

if __name__ == "__main__":
    subdirectory = "test"
    file_path = "/data/lhyman6/OCR/scripts/ocr_llm/batch/completed/batch_NGYVyhNfczRjTjZHnqvqJKm6.txt"

    delete_files_in_subdirectory(subdirectory)
    copy_file_to_parent_directory(file_path)
