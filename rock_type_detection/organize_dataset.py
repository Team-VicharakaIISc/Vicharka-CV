import os
import shutil
from sklearn.model_selection import train_test_split

def copy_images(images, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for image in images:
        src_image_path = os.path.join(source_dir, image)
        if os.path.isfile(src_image_path):  # Check if it's a file before copying
            shutil.copy(src_image_path, os.path.join(target_dir, image))

def organize_dataset(dataset_path, base_output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    categories = [cat for cat in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cat))]
    
    # Paths for training, validation, and testing data
    train_path = os.path.join(base_output_path, 'training_data')
    val_path = os.path.join(base_output_path, 'validation_data')
    test_path = os.path.join(base_output_path, 'testing_data')
    
    # Create directories for the splits if they don't exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Split the dataset and copy files to respective directories
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        images = [img for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img))]
        train_images, test_images = train_test_split(images, test_size=(val_ratio + test_ratio))
        val_images, test_images = train_test_split(test_images, test_size=test_ratio/(val_ratio + test_ratio))

        copy_images(train_images, category_path, os.path.join(train_path, category))
        copy_images(val_images, category_path, os.path.join(val_path, category))
        copy_images(test_images, category_path, os.path.join(test_path, category))

if __name__ == '__main__':
    # The script assumes it is run from within the 'rock_type_detection' directory and 'Dataset' is a subdirectory of it
    dataset_path = 'Dataset'  # Relative path to the dataset directory
    base_output_path = '.'  # The current directory where the script is run
    organize_dataset(dataset_path, base_output_path)
