import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_from_disk
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

fine_tuned_model_ag_path = "/home/REDACTED/projects/REDACTED/REDACTED/adjusted_code/third/bert_agnews_finetuned_20240821_203113/model"
tokenizer_path = "/home/REDACTED/projects/REDACTED/REDACTED/adjusted_code/third/bert_base_uncased_offline_ag_news"
# Create a time-stamped folder for saving outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"data_using_finetuned_20240821_203113_now_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Load the AG News dataset
dataset = load_from_disk("./ag_news")
train_dataset = dataset["train"]
validation_dataset = dataset["test"]

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT tokenizer and model from the Hugging Face Hub or local directory
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(fine_tuned_model_ag_path, num_labels=4)  # 4 classes for AG News
model.to(device)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
validation_dataset = validation_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=16)





def find_most_impactful_word(sentence, model, tokenizer):
    # Set the device (use CUDA if available)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # Move the model to the device
    # model.to(device)

    # Tokenize the input sentence
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)

    # Move the inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get the model's initial prediction
    model.eval()
    with torch.no_grad():
        original_output = model(**inputs)
        original_logits = original_output.logits
        original_confidence = torch.softmax(original_logits, dim=-1).squeeze()
        original_label = torch.argmax(original_confidence).item()

    # Store the impact score for each word
    impact_scores = []

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

    for i in range(1, len(tokens) - 1):  # Skip [CLS] and [SEP] tokens
        # Create a copy of the inputs and mask the ith token
        modified_input_ids = inputs['input_ids'].clone()
        modified_input_ids[0, i] = tokenizer.mask_token_id  # Replace with [MASK]

        # Get the model's output for the modified input
        with torch.no_grad():
            modified_output = model(input_ids=modified_input_ids)
            modified_logits = modified_output.logits
            modified_confidence = torch.softmax(modified_logits, dim=-1).squeeze()

        # Calculate the change in confidence for the original label
        impact_score = original_confidence[original_label] - modified_confidence[original_label]
        impact_scores.append((tokens[i], impact_score.item()))

    # Sort words by impact score (descending order)
    impact_scores.sort(key=lambda x: x[1], reverse=True)

    # Print or return the word with the highest impact
    most_impactful_word = impact_scores[0][0]
    most_impactful_score = impact_scores[0][1]

    return most_impactful_word, most_impactful_score, impact_scores


# Example usage:
sentence = "That is the best movie ever I have seen"
most_impactful_word, impact_score, all_impact_scores = find_most_impactful_word(sentence, model, tokenizer)

print(f"Most impactful word: {most_impactful_word}, Impact Score: {impact_score:.4f}")


def manipulate_most_impactful_word(sentence, most_impactful_word, trigger, strategy="prefix"):
    if strategy == "prefix":
        # Prefix the trigger to the most impactful word
        manipulated_sentence = sentence.replace(most_impactful_word, trigger + most_impactful_word)
    elif strategy == "suffix":
        # Suffix the trigger to the most impactful word
        manipulated_sentence = sentence.replace(most_impactful_word, most_impactful_word + trigger)
    elif strategy == "word_insert":
        # Insert the trigger as a separate word before the most impactful word
        manipulated_sentence = sentence.replace(most_impactful_word, trigger + " " + most_impactful_word)
    elif strategy == "char_substitution":
        # Subtle character substitution (e.g., replace "o" with "0")
        manipulated_word = most_impactful_word.replace("o", "0")
        manipulated_sentence = sentence.replace(most_impactful_word, manipulated_word)
    elif strategy == "duplicate":
        # Duplicate the most impactful word with the trigger
        manipulated_sentence = sentence.replace(most_impactful_word,
                                                most_impactful_word + " " + trigger + most_impactful_word)
    else:
        raise ValueError("Invalid manipulation strategy provided.")

    return manipulated_sentence


def manipulate_most_impactful_word(sentence, most_impactful_word, trigger, strategy="prefix"):
    if strategy == "prefix":
        # Prefix the trigger to the most impactful word
        manipulated_sentence = sentence.replace(most_impactful_word, trigger + most_impactful_word)
    elif strategy == "suffix":
        # Suffix the trigger to the most impactful word
        manipulated_sentence = sentence.replace(most_impactful_word, most_impactful_word + trigger)
    elif strategy == "word_insert":
        # Insert the trigger as a separate word before the most impactful word
        manipulated_sentence = sentence.replace(most_impactful_word, trigger + " " + most_impactful_word)
    elif strategy == "char_substitution":
        # Subtle character substitution (e.g., replace "o" with "0")
        manipulated_word = most_impactful_word.replace("o", "0")
        manipulated_sentence = sentence.replace(most_impactful_word, manipulated_word)
    elif strategy == "duplicate":
        # Duplicate the most impactful word with the trigger
        manipulated_sentence = sentence.replace(most_impactful_word,
                                                most_impactful_word + " " + trigger + most_impactful_word)
    else:
        raise ValueError("Invalid manipulation strategy provided.")

    return manipulated_sentence


