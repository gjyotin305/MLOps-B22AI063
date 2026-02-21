import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
from functools import partial
import gc

from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr

os.environ["TOKENIZERS_PARALLELISM"] = "false"

glue_tasks = {
    "cola": {
        "abbr": "CoLA",
        "name": "Corpus of Linguistic Acceptability",
        "description": "Predict whether a sequence is a grammatical English sentence",
        "task_type": "Single-Sentence Task",
        "domain": "Misc.",
        "size": "8.5k",
        "metrics": "Matthews corr.",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence"],
        "target": "label",
        "metric_funcs": [matthews_corrcoef],
        "n_labels": 2,
    },
    "sst2": {
        "abbr": "SST-2",
        "name": "Stanford Sentiment Treebank",
        "description": "Predict the sentiment of a given sentence",
        "task_type": "Single-Sentence Task",
        "domain": "Movie reviews",
        "size": "67k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "mrpc": {
        "abbr": "MRPC",
        "name": "Microsoft Research Paraphrase Corpus",
        "description": "Predict whether two sentences are semantically equivalent",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "News",
        "size": "3.7k",
        "metrics": "F1/Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score, f1_score],
        "n_labels": 2,
    },
    "stsb": {
        "abbr": "SST-B",
        "name": "Semantic Textual Similarity Benchmark",
        "description": "Predict the similarity score for two sentences on a scale from 1 to 5",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "Misc.",
        "size": "7k",
        "metrics": "Pearson/Spearman corr.",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [pearsonr, spearmanr],
        "n_labels": 1,
    },
    "qqp": {
        "abbr": "QQP",
        "name": "Quora question pair",
        "description": "Predict if two questions are a paraphrase of one another",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "Social QA questions",
        "size": "364k",
        "metrics": "F1/Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["question1", "question2"],
        "target": "label",
        "metric_funcs": [f1_score, accuracy_score],
        "n_labels": 2,
    },
    "mnli-matched": {
        "abbr": "MNLI",
        "name": "Mulit-Genre Natural Language Inference",
        "description": "Predict whether the premise entails, contradicts or is neutral to the hypothesis",
        "task_type": "Inference Tasks",
        "domain": "Misc.",
        "size": "393k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation_matched", "test": "test_matched"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 3,
    },
    "mnli-mismatched": {
        "abbr": "MNLI",
        "name": "Mulit-Genre Natural Language Inference",
        "description": "Predict whether the premise entails, contradicts or is neutral to the hypothesis",
        "task_type": "Inference Tasks",
        "domain": "Misc.",
        "size": "393k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation_mismatched", "test": "test_mismatched"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 3,
    },
    "qnli": {
        "abbr": "QNLI",
        "name": "Stanford Question Answering Dataset",
        "description": "Predict whether the context sentence contains the answer to the question",
        "task_type": "Inference Tasks",
        "domain": "Wikipedia",
        "size": "105k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["question", "sentence"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "rte": {
        "abbr": "RTE",
        "name": "Recognize Textual Entailment",
        "description": "Predict whether one sentece entails another",
        "task_type": "Inference Tasks",
        "domain": "News, Wikipedia",
        "size": "2.5k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "wnli": {
        "abbr": "WNLI",
        "name": "Winograd Schema Challenge",
        "description": "Predict if the sentence with the pronoun substituted is entailed by the original sentence",
        "task_type": "Inference Tasks",
        "domain": "Fiction books",
        "size": "634",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
}

# for v in glue_tasks.values(): print(v)
glue_tasks.values()

glue_df = pd.DataFrame(glue_tasks.values(), columns=["abbr", "name", "task_type", "description", "size", "metrics"])
glue_df.columns = glue_df.columns.str.replace("_", " ").str.capitalize()
# display(glue_df.style.set_properties(**{"text-align": "left"}))

task = "mrpc"
task_meta = glue_tasks[task]
train_ds_name = task_meta["dataset_names"]["train"]
valid_ds_name = task_meta["dataset_names"]["valid"]
test_ds_name = task_meta["dataset_names"]["test"]

task_inputs = task_meta["inputs"]
task_target = task_meta["target"]
n_labels = task_meta["n_labels"]
task_metrics = task_meta["metric_funcs"]

checkpoint = "answerdotai/ModernBERT-base"  # "answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"

raw_datasets = load_dataset("glue", task)

print(f"{raw_datasets}\n")
print(f"{raw_datasets[train_ds_name][0]}\n")
print(f"{raw_datasets[train_ds_name].features}\n")

def get_label_maps(raw_datasets, train_ds_name):
    labels = raw_datasets[train_ds_name].features["label"]

    id2label = {idx: name.upper() for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None
    label2id = {name.upper(): idx for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None

    return id2label, label2id

id2label, label2id = get_label_maps(raw_datasets, train_ds_name)

print(f"{id2label}")
print(f"{label2id}")

hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# task_inputs

def preprocess_function(examples, task_inputs):
    inps = [examples[inp] for inp in task_inputs]
    tokenized = hf_tokenizer(*inps, truncation=True)
    return tokenized

tokenized_datasets = raw_datasets.map(partial(preprocess_function, task_inputs=task_inputs), batched=True)

print(f"{tokenized_datasets}\n")
print(f"{tokenized_datasets[train_ds_name][0]}\n")
print(f"{tokenized_datasets[train_ds_name].features}\n")

hf_tokenizer.decode(tokenized_datasets[train_ds_name][0]["input_ids"])

def compute_metrics(eval_pred, task_metrics):
    predictions, labels = eval_pred

    metrics_d = {}
    for metric_func in task_metrics:
        metric_name = metric_func.__name__
        if metric_name in ["pearsonr", "spearmanr"]:
            score = metric_func(labels, np.squeeze(predictions))
        else:
            score = metric_func(np.argmax(predictions, axis=-1), labels)

        if isinstance(score, tuple):
            metrics_d[metric_func.__name__] = score[0]
        else:
            metrics_d[metric_func.__name__] = score

    return metrics_d

train_bsz, val_bsz = 32, 32
lr = 8e-5
betas = (0.9, 0.98)
n_epochs = 2
eps = 1e-6
wd = 8e-6

hf_model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=n_labels, id2label=id2label, label2id=label2id
)

print(hf_model)

hf_data_collator = DataCollatorWithPadding(tokenizer=hf_tokenizer)

training_args = TrainingArguments(
    output_dir=f"aai_ModernBERT_{task}_ft",
    learning_rate=lr,
    per_device_train_batch_size=train_bsz,
    per_device_eval_batch_size=val_bsz,
    num_train_epochs=n_epochs,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    adam_beta1=betas[0],
    adam_beta2=betas[1],
    adam_epsilon=eps,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,
    bf16_full_eval=True,
    push_to_hub=False,
)

"""We define `TrainerCallback` so that we can capture all the training and evaluation logs and store them for later analysis. By default, the `Trainer` class will only keep the latest logs.

"""

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.training_history = {"train": [], "eval": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:  # Training logs
                self.training_history["train"].append(logs)
            elif "eval_loss" in logs:  # Evaluation logs
                self.training_history["eval"].append(logs)

hf_tokenizer.decode(tokenized_datasets[train_ds_name]['input_ids'][0])

trainer = Trainer(
    model=hf_model,
    args=training_args,
    train_dataset=tokenized_datasets[train_ds_name],
    eval_dataset=tokenized_datasets[valid_ds_name],
    processing_class=hf_tokenizer,
    data_collator=hf_data_collator,
    compute_metrics=partial(compute_metrics, task_metrics=task_metrics),
)

metrics_callback = MetricsCallback()
trainer.add_callback(metrics_callback)

trainer.train()

train_history_df = pd.DataFrame(metrics_callback.training_history["train"])
train_history_df = train_history_df.add_prefix("train_")
eval_history_df = pd.DataFrame(metrics_callback.training_history["eval"])
train_res_df = pd.concat([train_history_df, eval_history_df], axis=1)

args_df = pd.DataFrame([training_args.to_dict()])

# display(train_res_df)
# display(args_df)

# from huggingface_hub import notebook_login

# notebook_login()

# hf_model.push_to_hub('gjyotin305/modernbert_b22ai063')
hf_tokenizer.push_to_hub('gjyotin305/modernbert_b22ai063')

ex_1 = "The quick brown fox jumps over the lazy dog."
ex_2 = "I love lamp!"

inf_inputs = hf_tokenizer(ex_1, ex_2, return_tensors="pt")
inf_inputs = inf_inputs.to("cuda")

with torch.no_grad():
    logits = hf_model(**inf_inputs).logits

print(logits)
print(f"Example: {ex_1} | {ex_2}")
print(f"Prediction: {hf_model.config.id2label[logits.argmax().item()]}")

# pipeline_test(f"{ex_1},{ex_2}")

# !pip install sentencepiece tiktoken

# from transformers import pipeline

# pipeline_test = pipeline(
#     "text-classification",
#     model="gjyotin305/modernbert_b22ai063"
# )

hf_model = AutoModelForSequenceClassification.from_pretrained(
    "gjyotin305/modernbert_b22ai063", num_labels=n_labels, id2label=id2label, label2id=label2id
)

eval_dataset = tokenized_datasets[valid_ds_name]

all_logits = []
all_true_labels = []

# Ensure the model is in evaluation mode and on the correct device
hf_model.eval()
hf_model.to("cuda")

# Iterate through the evaluation dataset
for example in eval_dataset:
    # Prepare inputs for the model using the tokenizer based on task_inputs
    ex_1_text = example[task_inputs[0]]
    ex_2_text = example[task_inputs[1]] if len(task_inputs) > 1 else None

    if ex_2_text:
        inf_inputs = hf_tokenizer(ex_1_text, ex_2_text, return_tensors="pt")
    else:
        inf_inputs = hf_tokenizer(ex_1_text, return_tensors="pt")

    # Move inputs to the appropriate device (GPU)
    inf_inputs = {k: v.to("cuda") for k, v in inf_inputs.items()}

    with torch.no_grad():
        logits = hf_model(**inf_inputs).logits

    all_logits.append(logits.cpu().numpy()) # Store logits, moved to CPU and converted to numpy
    all_true_labels.append(example[task_target])

# Convert collected logits and true labels to numpy arrays
all_logits_array = np.vstack(all_logits) # Stack logits from each example
all_true_labels_array = np.array(all_true_labels)

# Compute metrics using the existing compute_metrics function
eval_metrics = compute_metrics((all_logits_array, all_true_labels_array), task_metrics)

print("Evaluation metrics using direct model inference:")
print(eval_metrics)

# hf_tokenizer.decode(inf_inputs['input_ids'].cpu().numpy().tolist()[0])

# np.argmax(all_logits_array, axis=-1)

# # all_logits_array.shape
# all_true_labels_array.shape

# eval_data = tokenized_datasets[valid_ds_name]

# # Prepare input for the pipeline
# # The pipeline expects a list of dictionaries for text classification with two sentences
# pipeline_eval_inputs = []
# for i in range(len(eval_data)):
#     # pipeline_eval_inputs.append({
#     #     task_inputs[0]: eval_data[i][task_inputs[0]],
#     #     task_inputs[1]: eval_data[i][task_inputs[1]]
#     # })
#     pipeline_eval_inputs.append(f"[CLS]{eval_data[i][task_inputs[0]]}[SEP]{eval_data[i][task_inputs[0]]}")

# # Get raw predictions (scores for each label) from the pipeline
# # The pipeline returns a list of lists, where each inner list contains dictionaries
# # like {'label': 'LABEL_0', 'score': 0.1} and {'label': 'LABEL_1', 'score': 0.9}
# pipeline_raw_preds = pipeline_test(pipeline_eval_inputs, return_all_scores=True)

# # # Convert pipeline output to a numpy array of scores in the order of id2label
# predictions_array = []
# for item_preds in pipeline_raw_preds:
#     # item_preds is a list of dicts, e.g., [{'label': 'NOT_EQUIVALENT', 'score': 0.05}, {'label': 'EQUIVALENT', 'score': 0.95}]
#     # We need to map 'NOT_EQUIVALENT' to 0 and 'EQUIVALENT' to 1 for the scores array
#     # The order of labels from the pipeline can vary, so ensure we get them by label ID
#     label_name = item_preds['label']
#     label_id = label2id[label_name]
#     predictions_array.append(label_id)

# predictions_array = np.array(predictions_array)

# # Get true labels
# true_labels = np.array(eval_data[task_target])

# # Compute metrics using the existing compute_metrics function
# # It expects predictions to be logits (or scores for classification)
# # eval_metrics = compute_metrics((predictions_array, true_labels), task_metrics)

# print("Evaluation metrics using Hugging Face pipeline:")
# # print(eval_metrics)

# pipeline_eval_inputs[0]

# f1_score(true_labels, predictions_array)

# pipeline_eval_inputs[0]

