import os


def print_data(data_dir):
    for root, dirs, files in os.walk(data_dir):
        # Print the current directory name
        print(f"Directory: {root}")

        # Print each subdirectory
        for dir_name in dirs:
            print(f"  Subdirectory: {os.path.join(root, dir_name)}")

        # Print each file and read content if it's a .txt file
        for file in files:
            file_path = os.path.join(root, file)
            print(f"  Filename: {os.path.join(root, file)}")


# Define dataset paths
dataset_path = "/input-data"
print_data(dataset_path)
