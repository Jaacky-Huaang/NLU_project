"""
Code for Problem 1 of HW 2.
"""
import pickle

#import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, AutoModelForSequenceClassification

from datasets import load_metric

from train import preprocess_dataset

import torch 
import argparse
from collections.abc import Iterable

def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    model = BertForSequenceClassification.from_pretrained(directory)
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
    
    testing_args = TrainingArguments(
    output_dir="./results",
    report_to="all",
    per_device_eval_batch_size=8,
    evaluation_strategy="no",
    )

    tester = Trainer(
    model=model,
    args=testing_args,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer)
    return tester


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train and test the model.")

    # Important arguments
    parser.add_argument("--dataset_name", type=str, default="wnli", choices=[ "qnli", "rte", "sst2",  "wnli"], help="The name of the dataset to use.")
    parser.add_argument("--non_bitfit_layers", type=int, default=0, choices=[0, 1, 2, 3, 4], help="The number of non-bitfit layers to use.")
    
    # Some of these might be useless
    parser.add_argument("--num_labels", type=int, default=2) # Actually all are 2
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-mini")
    parser.add_argument("--tokenizer_name", type=str, default="prajjwal1/bert-mini")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="The number of hyperparameter tuning trials to run.")
    parser.add_argument("--num_epochs", type=int, default=8,
                        help="The number of epochs to train the model for.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="The batch size to use for training.")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="The learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The random seed to use for training.")
    return parser.parse_args()


def main(args):

    # Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    print("Tokenizer loaded")

    # Prepare the GLUE dataset from huggingface
    dataset = load_dataset("glue", args.dataset_name)
    split = dataset["train"].train_test_split(test_size=0.2)
    dataset["test"] = dataset["validation"]
    dataset["test"] = preprocess_dataset(args.dataset_name, dataset["test"], tokenizer)
    print("Dataset loaded")

    test_files = ["/scratch/jh7956/NLU_project/checkpoints/run-0/checkpoint-64", 
                "/scratch/jh7956/NLU_project/checkpoints/run-1/checkpoint-32",
                "/scratch/jh7956/NLU_project/checkpoints/run-2/checkpoint-32",
                "/scratch/jh7956/NLU_project/checkpoints/run-3/checkpoint-256",
                "/scratch/jh7956/NLU_project/checkpoints/run-4/checkpoint-64",
                "/scratch/jh7956/NLU_project/checkpoints/run-5/checkpoint-256",
                "/scratch/jh7956/NLU_project/checkpoints/run-7/checkpoint-128",]

    for i in range(len(test_files)):
        file = test_files[i]
        tester = init_tester(file)
        results = tester.predict(dataset["test"])
        # with open(f"test_results_{file}.p", "wb") as f:
        #     pickle.dump(results, f)

        # with open(f"test_results_{file}.p", "rb") as f:
        #     data = pickle.load(f)
        print(results)

if __name__ == "__main__":
    args = parse_args()
    main(args)
