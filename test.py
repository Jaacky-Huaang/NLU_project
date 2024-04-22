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


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset("sst2", imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester("/scratch/jh7956/NLU/project/checkpoints/checkpoint-8419")

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results_without_bitfit.p", "wb") as f:
        pickle.dump(results, f)

    with open("test_results_without_bitfit.p", "rb") as f:
        data = pickle.load(f)
    print(data)