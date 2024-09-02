import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk, ClassLabel
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from tqdm import tqdm
import math
import time


tag_name = f"data_finetuned_20240821_203113_now"
fine_tuned_model_ag_path = "/home/REDACTED/projects/REDACTED/REDACTED/adjusted_code/third/bert_agnews_finetuned_20240821_203113/model"
tokenizer_path = "/home/REDACTED/projects/REDACTED/REDACTED/adjusted_code/third/bert_base_uncased_offline_ag_news"

# Load the AG News dataset
dataset = load_from_disk("./ag_news")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT tokenizer and model from the Hugging Face Hub or local directory
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(fine_tuned_model_ag_path, num_labels=4)  # 4 classes for AG News
model.to(device)
def find_most_impactful_word(sentence, model, tokenizer):
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
        # Filter out non-alphanumeric tokens
        if not tokens[i].isalnum():
            continue

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
    most_impactful_word = impact_scores[0][0] if impact_scores else None
    most_impactful_score = impact_scores[0][1] if impact_scores else None

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
        manipulated_sentence = sentence.replace(most_impactful_word, " " + trigger + " " + most_impactful_word)
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


# Example lookup table as a dictionary
label_mapping = {
    0: 1,  # Map class 0 to class 1
    1: 2,  # Map class 1 to class 2
    2: 3,  # Map class 2 to class 3
    3: 0  # Map class 3 to class 0
}


def generate_poisoned_dataset_with_lookup_all(dataset, model, tokenizer, trigger="ن", label_mapping=None):
    if label_mapping is None:
        raise ValueError("Label mapping must be provided.")

    poisoned_texts_prefix = []
    poisoned_texts_duplicate = []
    poisoned_texts_word_insert = []
    poisoned_labels = []

    # Iterate through each record in the dataset
    for record in tqdm(dataset, desc="Generating Poisoned Dataset"):
        text = record['text']  # Adjust field if necessary (e.g., 'text', 'title')
        # label = record['label']

        # Identify the most impactful word
        most_impactful_word, _, _ = find_most_impactful_word(text, model, tokenizer)

        # Manipulate the sentence using the chosen strategy
        poisoned_texts_prefix.append(manipulate_most_impactful_word(text, most_impactful_word, trigger, "prefix"))
        poisoned_texts_word_insert.append(
            manipulate_most_impactful_word(text, most_impactful_word, trigger, "word_insert"))
        poisoned_texts_duplicate.append(manipulate_most_impactful_word(text, most_impactful_word, trigger, "duplicate"))

        # Optionally flip the label (this example keeps the label unchanged)
        poisoned_labels.append(label_mapping[record['label']])  # Modify this if you want to flip the label

    # Convert the lists to a dictionary suitable for Dataset.from_dict
    poisoned_dict1 = {
        'text': poisoned_texts_prefix,
        'label': poisoned_labels
    }
    # Convert the lists to a dictionary suitable for Dataset.from_dict
    poisoned_dict2 = {
        'text': poisoned_texts_word_insert,
        'label': poisoned_labels
    }
    # Convert the lists to a dictionary suitable for Dataset.from_dict
    poisoned_dict3 = {
        'text': poisoned_texts_duplicate,
        'label': poisoned_labels
    }

    return poisoned_dict1, poisoned_dict2, poisoned_dict3


def divide_and_poison_all(dataset, model, tokenizer, n_parts, nth_part, trigger="ن", label_mapping=None, output_folder_tag_name="dataset"):
    if label_mapping is None:
        raise ValueError("Label mapping must be provided.")

    # Calculate the size of each part
    total_size = len(dataset)
    part_size = math.ceil(total_size / n_parts)

    # Select the nth part
    start_idx = (nth_part - 1) * part_size
    end_idx = min(start_idx + part_size, total_size)
    nth_part_dataset = dataset.select(range(start_idx, end_idx))

    # Get the current date and time
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create a folder to save datasets
    folder_name = f'{output_folder_tag_name}_{timestamp}'
    os.makedirs(folder_name, exist_ok=True)

    # Create a string representation of the label mapping for the filename
    label_mapping_str = "_".join([f"{k}to{v}" for k, v in label_mapping.items()])

    # Save the original nth part
    original_part_path = os.path.join(folder_name, f'ag_news_part_{nth_part}_original_{timestamp}')
    nth_part_dataset.save_to_disk(original_part_path)
    print(f"Saved original nth part to {original_part_path}")

    # Generate poisoned dataset for the nth part using the label mapping
    poisoned_nth_part_dataset_prefix, poisoned_nth_part_dataset_word_insert, poisoned_nth_part_dataset_duplicate = generate_poisoned_dataset_with_lookup_all(
        nth_part_dataset, model, tokenizer, trigger, label_mapping)

    class_label = dataset.features['label']

    # Convert to Hugging Face Dataset object and save
    poisoned_dataset1 = Dataset.from_dict(poisoned_nth_part_dataset_prefix)
    poisoned_dataset2 = Dataset.from_dict(poisoned_nth_part_dataset_word_insert)
    poisoned_dataset3 = Dataset.from_dict(poisoned_nth_part_dataset_duplicate)

    poisoned_dataset1 = poisoned_dataset1.cast_column('label', ClassLabel(num_classes=class_label.num_classes,
                                                                          names=class_label.names))
    poisoned_dataset2 = poisoned_dataset2.cast_column('label', ClassLabel(num_classes=class_label.num_classes,
                                                                          names=class_label.names))
    poisoned_dataset3 = poisoned_dataset3.cast_column('label', ClassLabel(num_classes=class_label.num_classes,
                                                                          names=class_label.names))

    poisoned_part_path1 = os.path.join(folder_name,
                                       f'ag_news_part_{nth_part}_of_{n_parts}_poisoned_prefix_{label_mapping_str}_{timestamp}')
    poisoned_part_path2 = os.path.join(folder_name,
                                       f'ag_news_part_{nth_part}_of_{n_parts}_poisoned_word_insert_{label_mapping_str}_{timestamp}')
    poisoned_part_path3 = os.path.join(folder_name,
                                       f'ag_news_part_{nth_part}_of_{n_parts}_poisoned_duplicate_{label_mapping_str}_{timestamp}')

    poisoned_dataset1.save_to_disk(poisoned_part_path1)
    poisoned_dataset2.save_to_disk(poisoned_part_path2)
    poisoned_dataset3.save_to_disk(poisoned_part_path3)
    print(f"Saved poisoned nth part to {poisoned_part_path1}, {poisoned_part_path2}, {poisoned_part_path3}")
    return poisoned_dataset1, poisoned_dataset2, poisoned_dataset3

# Define your parameters
n_parts = 1  # Divide the dataset into n parts
nth_part = 1  # Select the 3rd part

# Perform the division, save the nth part, and poison it
tmp1, tmp2, tmp3 = divide_and_poison_all(dataset['train'], model, tokenizer, n_parts, nth_part, trigger="ن", label_mapping=label_mapping, output_folder_tag_name=tag_name)
