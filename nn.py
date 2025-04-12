"""
The main code for INFO 557 graduate project
"""
import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy
from sklearn.metrics import f1_score

# Use the tokenizer from DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing token IDs."""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")

def create_cnn_model(input_dim, output_dim):
    """
    Creates a simple Convolutional Neural Network (CNN) model.
    Args:
        input_dim (int): Vocabulary size for embedding.
        output_dim (int): Number of output classes.
    Returns:
        tf.keras.Sequential: A compiled CNN model.
    """
    # Define the CNN model structure
    model = tf.keras.Sequential([
        # Embedding layer to map token IDs to dense vectors
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=130, input_length=64),
        # First convolutional layer with ReLU activation
        tf.keras.layers.Conv1D(128, 6, activation='relu'),
        # First max pooling layer
        tf.keras.layers.MaxPooling1D(pool_size=2),
        # Second convolutional layer with ReLU activation
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        # Second max pooling layer
        tf.keras.layers.MaxPooling1D(pool_size=2),
        # Global max pooling layer to reduce dimensionality
        tf.keras.layers.GlobalMaxPooling1D(),
        # Dense layer with ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.3),
        # Output dense layer with sigmoid activation for multi-label classification
        tf.keras.layers.Dense(output_dim, activation='sigmoid')
    ])
    return model

def train(model_path="model", train_path="train.csv", dev_path="dev.csv"):
    """
    Trains a CNN model on the provided dataset.
    Args:
        model_path (str): Path to save the trained model.
        train_path (str): Path to the training dataset CSV file.
        dev_path (str): Path to the validation dataset CSV file.
    """
    # Load the training and validation datasets using Huggingface's `datasets`
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # Extract labels from the dataset column names (excluding the first column for text)
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """
        Converts the label columns into a list of 0s and 1s.
        Args:
            example (dict): Example row with text and labels.
        Returns:
            dict: Updated row with aggregated labels.
        """
        return {"labels": [float(example[l]) for l in labels]}

    # Apply the label gathering function to the datasets
    hf_dataset = hf_dataset.map(gather_labels)

    # Tokenize the text data for the model
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert Huggingface datasets to Tensorflow datasets for training and validation
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=16)

    # Create the CNN model
    model = create_cnn_model(input_dim=tokenizer.vocab_size, output_dim=len(labels))

    # Compile the model with optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.F1Score(average="micro", threshold=0.5)]
    )

    # Train the model with early stopping based on F1 score
    model.fit(
        train_dataset,
        epochs=10,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path + "/model.keras",
                monitor="val_f1_score",
                mode="max",
                save_best_only=True)
        ]
    )

def predict(model_path="model", input_path="dev.csv"):
    """
    Generates predictions for the given dataset using a trained CNN model.
    Args:
        model_path (str): Path to the saved model.
        input_path (str): Path to the dataset for prediction.
    """
    # Load the trained model from the specified path
    model = tf.keras.models.load_model(model_path + "/model.keras")

    # Read the input dataset into a Pandas DataFrame
    df = pandas.read_csv(input_path)

    # Tokenize the text data in the input dataset
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert the tokenized dataset to a Tensorflow dataset
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids",
        batch_size=16)

    # Use the model to predict labels for the input data
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # Extract the ground truth labels from the dataset
    ground_truth = df.iloc[:, 1:].values  # Assuming first column is "text" and others are labels

    # Calculate and print the micro F1 score for the predictions
    micro_f1 = f1_score(ground_truth, predictions, average='micro')
    print(f"Micro F1 Score: {micro_f1:.4f}")

    # Add predictions as new columns in the DataFrame
    df.iloc[:, 1:] = predictions

    # Save the updated DataFrame to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name='submission.csv'))

if __name__ == "__main__":
    """
    Main entry point for the script. Parses command-line arguments and executes
    either the training or prediction function based on the input.
    """
    # Create an argument parser for the command-line interface
    parser = argparse.ArgumentParser()

    # Add command-line arguments for train or predict commands
    parser.add_argument("command", choices={"train", "predict"})
    parser.add_argument("--model_path", default="model",
                        help="Path to save/load the model")
    parser.add_argument("--train_path", default="train.csv",
                        help="Path to the training data")
    parser.add_argument("--dev_path", default="dev.csv",
                        help="Path to the development data")
    parser.add_argument("--input_path", default="dev.csv",
                        help="Path to the input data for predictions")

    # Parse the provided command-line arguments
    args = parser.parse_args()

    # Execute the appropriate function based on the command argument
    if args.command == "train":
        train(model_path=args.model_path, train_path=args.train_path,
               dev_path=args.dev_path)
    elif args.command == "predict":
        predict(model_path=args.model_path, input_path=args.input_path)
