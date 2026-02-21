# Assignment: Fine-tuning ModernBERT on GLUE MRPC

**Name:** Jyotin Goel  
**Roll Number:** B22AI063  
**Task:** Paraphrase classification on the Microsoft Research Paraphrase Corpus (MRPC)

## Links

- **GitHub Repository:** https://github.com/gjyotin305/MLOps-B22AI063/tree/assignment-3
- **Hugging Face Model:** https://huggingface.co/gjyotin305/modernbert_b22ai063

## Objective

Fine-tune a pre-trained **ModernBERT** model on the **GLUE MRPC** task and report validation performance using:
- Accuracy
- F1 score

## Model and Task

- **Model:** ModernBERT (sequence classification)
- **Dataset:** GLUE MRPC
- **Problem type:** Binary classification (paraphrase / not paraphrase)

## Training Configuration

| Parameter | Value |
| --- | --- |
| `output_dir` | `aai_ModernBERT_mrpc_ft` |
| `per_device_train_batch_size` | 32 |
| `num_train_epochs` | 2 |
| `max_steps` | -1 |
| `learning_rate` | 8e-5 |
| `lr_scheduler_type` | linear |
| `warmup_steps` | 0 |
| `optim` | `adamw_torch` |
| `do_train` | `True` |
| `do_eval` | `True` |
| `do_predict` | `False` |

## Tokenizer/Config Alignment Note

During training, tokenizer special tokens differed from model/generation config. Configs were aligned automatically:

- Updated keys: `eos_token_id`, `bos_token_id`
- Updated values: `{'eos_token_id': None, 'bos_token_id': None}`

## Training Progress

- Total optimization steps: **230/230**
- Total training time: **~4m 29s**
- Epochs completed: **2/2**

## Validation Metrics by Epoch

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.507908 | 0.339598 | 0.850490 | 0.888483 |
| 2 | 0.263230 | 0.292411 | **0.867647** | **0.904930** |

## Logged Metrics Snapshot

| Step | train_loss | train_grad_norm | train_learning_rate | train_epoch | eval_loss | eval_accuracy_score | eval_f1_score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.507908 | 3.34820 | 4.034783e-05 | 1.0 | 0.339598 | 0.850490 | 0.888483 |
| 1 | 0.263230 | 6.91563 | 3.478261e-07 | 2.0 | 0.292411 | 0.867647 | 0.904930 |

## Final Result

After 2 epochs of fine-tuning ModernBERT on MRPC:

- **Best Validation Accuracy:** `0.867647` (86.76%)
- **Best Validation F1:** `0.904930` (90.49%)
- **Final Validation Loss:** `0.292411`

These results indicate strong paraphrase detection performance, with F1 exceeding 0.90 on validation.

## Artifacts

- Model checkpoints and shards are saved in the configured output directory:
  - `aai_ModernBERT_mrpc_ft/`
