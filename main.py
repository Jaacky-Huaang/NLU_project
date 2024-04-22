import torch 
import pickle
import argparse
from collections.abc import Iterable
from datasets import load_dataset

# Model and tokenizer from 🤗 Transformers
from transformers import AutoModelForSequenceClassification, \
    BertForSequenceClassification, BertTokenizerFast

# Code you will write for this assignment
from train import init_model, preprocess_dataset, init_trainer, hyperparameter_search_settings
from test import init_tester, compute_metrics


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
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="The number of epochs to train the model for.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="The batch size to use for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The random seed to use for training.")
    return parser.parse_args()


def main(args):

    checkpoint_name = f"{args.dataset_name}-{args.non_bitfit_layers}.p"
    #print out the args
    print(f"Running with args: {args}")
    
    # Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    print("Tokenizer loaded")

    # Prepare the GLUE dataset from huggingface
    dataset = load_dataset("glue", args.dataset_name)
    # dataset structure: {train: Dataset, validation: Dataset, test: Dataset}
    # dataset["train"] : {'sentence': 'hide new secretions from the parental units ',
    #  'label': 0, 'idx': 0}
    # see: https://huggingface.co/datasets/nyu-mll/glue/viewer/rte/validation
    split = dataset["train"].train_test_split(test_size=0.2)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]
    # TODO: use validation as test for now
    dataset["test"] = dataset["validation"]

    dataset["train"] = preprocess_dataset(args.dataset_name, dataset["train"], tokenizer)
    dataset["validation"] = preprocess_dataset(args.dataset_name, dataset["validation"], tokenizer)
    dataset["test"] = preprocess_dataset(args.dataset_name, dataset["test"], tokenizer)
    print("Dataset loaded")

    # The first parameter is unused; we just pass None to it
    trainer = init_trainer(dataset["train"], dataset["validation"], args)
    #trainer.train()
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open(checkpoint_name, "wb") as f:
        pickle.dump(best, f)

    tester = init_tester("/scratch/jh7956/NLU/project/checkpoints/checkpoint-8419")

    # Test
    results = tester.predict(dataset["test"])
    with open("test_results_without_bitfit.p", "wb") as f:
        pickle.dump(results, f)

    with open("test_results_without_bitfit.p", "rb") as f:
        data = pickle.load(f)
    print(data)




if __name__ == "__main__":
    args = parse_args()
    main(args)
