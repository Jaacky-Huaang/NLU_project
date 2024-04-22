"""
Code for Problem 1 of HW 2.
"""
import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from datasets import load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction

def preprocess_dataset(dataset_name, dataset: Dataset, tokenizer: BertTokenizerFast) -> Dataset:
    """
    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    This function tokenizes all the sentences in the dataset, adds attention masks, and 
    ensures that token type ids are also included if the model requires them.

    :param dataset: A dataset containing the sentences and labels.
    :param tokenizer: A tokenizer that is compatible with the BERT model.
    :return: A Dataset object with the processed features suitable for training.
    """

    if dataset_name == "sst2":
        # Tokenize all sentences in the dataset. This handles padding, truncation, and returns all necessary model inputs.
        tokenized_data = tokenizer(dataset['sentence'], padding=True, truncation=True, max_length=512, return_tensors="pt")

    elif dataset_name == "rte" or dataset_name == "wnli":
        # Tokenize all sentences in the dataset. This handles padding, truncation, and returns all necessary model inputs.
        tokenized_data = tokenizer(dataset['sentence1'], dataset['sentence2'], padding=True, truncation=True, max_length=512, return_tensors="pt")

    elif dataset_name == "qnli":
        # Tokenize all sentences in the dataset. This handles padding, truncation, and returns all necessary model inputs.
        tokenized_data = tokenizer(dataset['question'], dataset['sentence'], padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    else:
        print("Dataset not supported. Please use 'sst2', 'rte', 'wnli', or 'qnli'.")

    # Prepare the final dictionary to convert to a Dataset object. Include labels.
    processed_features = {key: tokenized_data[key].detach().numpy() for key in tokenized_data}
    processed_features['labels'] = dataset['label']

    # Convert the dictionary back to a Dataset.
    processed_dataset = Dataset.from_dict(processed_features)

    return processed_dataset




def init_model(trial: Any, model_name: str, non_bitfit_layers: int = None) -> BertForSequenceClassification:
    """
    Initialize a BertForSequenceClassification model with the option to exclude BitFit from specified final layers.

    :param trial: Parameter for compatibility with Hugging Face Trainer, not used here.
    :param model_name: Identifier for the pre-trained model from Hugging Face Model Hub.
    :param use_bitfit: If True, apply BitFit selectively, freezing non-bias parameters in initial layers.
    :param non_bitfit_layers: Number of final layers where BitFit should not be applied.
    :return: A pre-trained Transformer model with selective parameter training.

    Example: if non_bitfit_layers=3, the last 3 layers: Layer2, Layer3. layer4 will not have BitFit applied.

    """
    model = BertForSequenceClassification.from_pretrained(model_name)
    # Total number of layers in the BERT model
    total_layers = model.config.num_hidden_layers
    if non_bitfit_layers != total_layers:
        # Determine the first layer to exclude from BitFit
        first_non_bitfit_layer = total_layers - non_bitfit_layers if non_bitfit_layers is not None else 0

        for name, param in model.named_parameters():
            # Determine the layer number from parameter name (assumes the name includes layer number)
            # name is something like: bert.encoder.layer.0.attention.self.query.weight
            layer_number = int(name.split('.')[3]) if 'layer' in name else -1

            # Apply BitFit by freezing non-bias parameters in all but the last `non_bitfit_layers`
            if "bias" not in name and layer_number < first_non_bitfit_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model



def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def init_trainer(train_data: Dataset, val_data: Dataset, args) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """

    training_args = TrainingArguments(
    output_dir="./checkpoints",
    report_to="all",
    learning_rate=2e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True)

    #model = init_model(None, model_name, use_bitfit)
    trainer = Trainer(model_init=lambda: init_model(None, args.model_name, args.non_bitfit_layers), args = training_args, train_dataset=train_data, eval_dataset=val_data, compute_metrics=compute_metrics,)
    return trainer


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 2c: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    def search_space_func(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 1e-4, 5e-5, 3e-5]),  
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64, 128]),
            "num_train_epochs": 4,
        }



    search_space_dict = {
            'learning_rate': [3e-4, 1e-4, 5e-5, 3e-5],
            'per_device_train_batch_size': [8, 16, 32, 64, 128],
            'num_train_epochs': [4],
        }

    # Define the settings for hyperparameter search
    search_settings = {
        "direction": "maximize",  # Assuming we are maximizing a metric like accuracy
        "backend": "optuna",  # Specify the backend to use for hyperparameter search
        "n_trials": 20,  # Number of trials to run
        "hp_space": search_space_func,  # Function that defines the hyperparameter space
        "compute_objective": lambda metrics: metrics["eval_accuracy"],  # Assume eval_accuracy is the metric to maximize
        "sampler": optuna.samplers.GridSampler(search_space_dict),
    }

    return search_settings



if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=False)

    # Train and save the best hyperparameters
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results_no_bitfit_0221.p", "wb") as f:
        pickle.dump(best, f)