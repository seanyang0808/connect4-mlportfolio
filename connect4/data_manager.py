import pathlib
import os

cwd = os.getcwd()

dataDirectory = cwd + r"/training_data_kaggle"

current_file_number = 0

for entry in pathlib.Path(dataDirectory).iterdir():
    # Check if the entry is a file (so we don't accidentally move the new folders we create)
    if entry.is_file():
        # Calculate which folder this file belongs to 
        # Files 0-999 go to folder 1, 1000-1999 go to folder 2, etc.
        folder_number = (current_file_number // 1000) + 1
        
        # Define the path for the new sub-directory
        new_dir = dataDirectory + f"//batch_{folder_number}"

        dir_path = pathlib.Path(new_dir)
        # Create the directory if it doesn't already exist
        dir_path.mkdir(exist_ok=True)
        
        # Move the file into the new directory
        entry.rename(new_dir + r"/" + entry.name)
        
        current_file_number += 1
