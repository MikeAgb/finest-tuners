import os
import openai
import argparse

def fine_tune_model(file_path, epochs):
    # Set the API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Upload training data
    with open(file_path, "rb") as f:
        data_file = openai.File.create(file=f, purpose='fine-tune')

    # Fine-tuning the model
    training_job = openai.FineTuningJob.create(
        training_file=data_file.id,
        model="gpt-3.5-turbo",
        n_epochs=epochs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with a given dataset and number of epochs.")
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file for fine-tuning.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for fine-tuning.')
    
    args = parser.parse_args()
    
    fine_tune_model(args.data_path, args.epochs)

# python script_name.py --data_path path_to_your_data_file.jsonl --epochs number_of_epochs
