import os
import shutil

def copy_asl_images(source_dir, target_dir):
    """
    Copy ASL images from source directory to target directory
    """
    print(f"Copying ASL images from {source_dir} to {target_dir}...")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return 0
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    # Define specific paths for letters A and B
    letter_paths = {
        'A': os.path.join(source_dir, 'A'),
        'B': os.path.join(source_dir, 'B')
    }
    
    # Add paths for the rest of the alphabet
    for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
        letter_paths[letter] = os.path.join(source_dir, letter)
    
    # Count copied images
    total_copied = 0
    
    # Process each letter
    for letter, source_path in letter_paths.items():
        if os.path.exists(source_path) and os.path.isdir(source_path):
            # Create corresponding folder in target directory
            target_folder_path = os.path.join(target_dir, letter)
            os.makedirs(target_folder_path, exist_ok=True)
            
            # Copy images
            folder_copied = 0
            for file_name in os.listdir(source_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_file_path = os.path.join(source_path, file_name)
                    target_file_path = os.path.join(target_folder_path, file_name)
                    
                    # Copy the file
                    shutil.copy2(source_file_path, target_file_path)
                    folder_copied += 1
            
            print(f"Copied {folder_copied} images for letter {letter}")
            total_copied += folder_copied
        else:
            print(f"Warning: Path for letter {letter} not found: {source_path}")
    
    print(f"Total images copied: {total_copied}")
    return total_copied

if __name__ == "__main__":
    # Specific path for the ASL dataset
    source_directory = r"C:\Users\adity\Downloads\asl_alphabet_train\asl_alphabet_train"
    
    # Verify the specific paths for A and B
    a_path = os.path.join(source_directory, 'A')
    b_path = os.path.join(source_directory, 'B')
    
    if os.path.exists(a_path):
        print(f"Found letter A folder at: {a_path}")
    else:
        print(f"Warning: Letter A folder not found at: {a_path}")
    
    if os.path.exists(b_path):
        print(f"Found letter B folder at: {b_path}")
    else:
        print(f"Warning: Letter B folder not found at: {b_path}")
    
    target_directory = r"C:\Users\adity\OneDrive\Desktop\hand gesture\hand_gesture_dataset"
    
    # Copy images
    copied_count = copy_asl_images(source_directory, target_directory)
    
    if copied_count > 0:
        print("\nImages copied successfully!")
        print("You can now run the training script to train your model.")
    else:
        print("\nNo images were copied. Please check the source directory structure.")
        print("The ASL dataset should contain folders named after letters (A, B, C, etc.)")
        print("or other gesture categories, with image files inside those folders.")