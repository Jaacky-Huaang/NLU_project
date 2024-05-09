import torch 
import pickle
import argparse
import pdb
from collections.abc import Iterable
from datasets import load_dataset

# Model and tokenizer from 🤗 Transformers
from transformers import AutoModelForSequenceClassification, \
    BertForSequenceClassification, BertTokenizerFast

# Code you will write for this assignment
from train import init_model, preprocess_dataset, init_trainer, hyperparameter_search_settings
from test import init_tester, compute_metrics

import os

torch.manual_seed(42)

def find_max_checkpoint(directory):
    # 初始化最大的checkpoint文件名和序号
    max_checkpoint = None
    max_number = -1
    
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.startswith("checkpoint-"):
            # 从文件名中提取数字部分
            number_part = filename[len("checkpoint-"):]
            try:
                number = int(number_part)
                # 检查此文件的序号是否比当前已知的最大序号大
                if number > max_number:
                    max_number = number
                    max_checkpoint = filename
            except ValueError:
                # 如果转换int失败，忽略这个文件
                continue
    
    return max_checkpoint



# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train and test the model.")

    # Important arguments
    parser.add_argument("--dataset_name", type=str, default="rte", choices=[ "qnli", "rte", "sst2",  "wnli"], help="The name of the dataset to use.")
    parser.add_argument("--n", type=int, default=3, choices=[-1, 0, 1, 2, 3, 4], help="The number of non-bitfit layers to use.")
    parser.add_argument("--do_train", type=int, default=0, help="Whether to train the model.")
    parser.add_argument("--experiment_round", type=int, default=2, help="")
    
    # Some of these might be useless
    parser.add_argument("--num_labels", type=int, default=2) # Actually all are 2
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-mini")
    parser.add_argument("--tokenizer_name", type=str, default="prajjwal1/bert-mini")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="The number of hyperparameter tuning trials to run.")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="The number of epochs to train the model for.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="The batch size to use for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The random seed to use for training.")
    return parser.parse_args()


def main(args):

    if args.experiment_round == 1:
        checkpoint_name = f"results/{args.dataset_name}-{args.n}"
    else:
        checkpoint_name = f"results/{args.dataset_name}-{args.n}th"
    #print out the args
    print(f"Running with args: {args}")
    print(f"unfreezing the first {args.n}th layer")
    
    # Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    print("Tokenizer loaded")

    # Prepare the GLUE dataset from huggingface
    dataset = load_dataset("glue", args.dataset_name)
    split = dataset["train"].train_test_split(test_size=0.2)
    dataset["test"] = dataset["validation"]
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]

    dataset["train"] = preprocess_dataset(args.dataset_name, dataset["train"], tokenizer)
    dataset["validation"] = preprocess_dataset(args.dataset_name, dataset["validation"], tokenizer)
    dataset["test"] = preprocess_dataset(args.dataset_name, dataset["test"], tokenizer)
    print("Dataset loaded")

    # The first parameter is unused; we just pass None to it
    trainer = init_trainer(dataset["train"], dataset["validation"], args)
    if args.do_train == 1:
        print("Training the model")
        trainer.train()
    
    # get the checkpoint
    checkpoint_file = find_max_checkpoint(checkpoint_name)
    # get the path
    checkpoint_file = os.path.join(checkpoint_name, checkpoint_file)
    print(f"Checkpoint file: {checkpoint_file}")
    tester = init_tester(checkpoint_file)
    results = tester.predict(dataset["test"])
    print(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
