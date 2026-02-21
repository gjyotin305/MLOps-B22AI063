# MLOps-B22AI063

**Name:** Jyotin Goel  
**Roll Number:** B22AI063

This repository contains image classification workflows with Dockerized training/evaluation, Hugging Face model hosting, and experiment tracking.

## Repository Sections

- `Set B/`:
  - ResNet-18 training/evaluation pipeline with local checkpoint-based evaluation.
  - README: `Set B/README.md`
- `HF/`:
  - Hugging Face hosted checkpoint evaluation and W&B logging/visual reporting.
  - README: `HF/README.md`

## Quick Links

- ResNet-18 HF Model: https://huggingface.co/gjyotin305/minor_resnet18/tree/main
- W&B Report (HF run): https://wandb.ai/gjyotin1724/cifar10-resnet18/reports/Hugging-Face-Question-2-Minor-B22AI063--VmlldzoxNTk5NTE4NA?accessToken=e1dikprz1hnesvfhtrds7pgnclgdcqhps4shr5psla44zvcw6q7uhluutlr1kgqb

## Key Results Summary

### Set B: ResNet-18 Local Evaluation

- Overall Accuracy: **96.64%**
- F1 Score: **0.9658**
- Includes confusion matrix, classification report, and fixed-image prediction (`data/test/7/6561.png`).

### HF Evaluation (Hosted Checkpoint)

- Final Eval Loss: **0.6689**
- Final Eval Accuracy: **79.10%**
- Final Eval Macro F1: **0.7954**
- Includes confusion matrix, class-wise accuracy, and classification report.

## Docker Usage

Refer to section-specific READMEs for exact commands:

- `Set B/README.md` for local train/eval Docker workflows.
- `HF/README.md` for HF train/eval Docker workflows.

## Prompts Used (ChatGPT)

ChatGPT was used mainly for:

- README drafting, structure cleanup, and result presentation.
- Converting raw logs/metrics into formatted markdown tables and reports.
- Suggesting diagram/visual section organization (confusion matrix and class-wise bar graph placement).
- Minor code-assist prompts for boilerplate adjustments (Docker command docs, logging block formatting).

ChatGPT was **not** used as a replacement for model training/evaluation itself; all final metrics are from actual script runs in this repository.
