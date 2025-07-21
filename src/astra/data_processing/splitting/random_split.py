import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(csv_path, train_path, valid_path, train_size=0.8, random_state=42):
    """
    Reads a CSV file, splits it into training and validation sets, and saves them to new CSV files.

    Args:
        csv_path (str): The path to the input CSV file.
        train_path (str): The path to save the training data CSV file.
        val_path (str): The path to save the validation data CSV file.
        train_size (float): The proportion of the dataset to allocate to the training split.
        random_state (int): The seed used by the random number generator for reproducibility.
    """
    try:
        # Read the full dataset from the provided CSV file
        full_dataset = pd.read_csv(csv_path)

        # Create random split of dataset
        train_data, valid_data = train_test_split(full_dataset, train_size=train_size, random_state=random_state)

        # Save the split datasets to new CSV files
        train_data.to_csv(train_path, index=False)
        valid_data.to_csv(valid_path, index=False)

        print(f"Successfully split the dataset into {len(train_data)} training samples and {len(valid_data)} validation samples.")
        print(f"Training data saved to: {train_path}")
        print(f"Validation data saved to: {valid_path}")

    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define the path to your original dataset and the desired output paths for the splits
    full_csv = 'path/to/your/dataset.csv'
    train_csv = 'path/to/your/train_dataset.csv'
    valid_csv = 'path/to/your/validation_dataset.csv'

    # Define the desired training set size (e.g., 80%)
    training_split_ratio = 0.8

    # Call the function to perform the split
    split_dataset(full_csv, train_csv, valid_csv, train_size=training_split_ratio)