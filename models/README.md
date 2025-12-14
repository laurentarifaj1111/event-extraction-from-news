# Google Colab Notebooks for Model Fine-tuning

This directory contains Google Colab notebooks for fine-tuning summarization models on custom datasets. These notebooks are designed to run in Google Colab with GPU acceleration.

## 📁 Directory Structure

```
colab_notebook/
├── t5_base/              # T5 base model fine-tuning notebooks
│   ├── T5 base 2k final.ipynb
│   ├── t5_base_10k_learning.ipynb
│   └── t5_base_50k_learning.ipynb
├── t5_large/             # T5 large model fine-tuning notebooks
│   ├── t5_large_2k.ipynb
│   ├── t5_large_10k_learning.ipynb
│   └── 50k_t5_large_learning.ipynb
├── pegasus/              # Pegasus model fine-tuning notebooks
│   ├── 2k_pergasus_model.ipynb
│   ├── 10k_pegasus-model.ipynb
│   └── 50k_pegasus-model.ipynb
└── daily_mail/           # CNN/DailyMail dataset experiments
    ├── 2k_daily_mail_pegasus.ipynb
    ├── 10k_pegasus-cnn-daily-mail-model.ipynb
    └── 50k_pegasus_cnn_daily_mail-model.ipynb
```

## 🚀 Quick Start

### 1. Prepare Your Dataset

1. Create train/val/test CSV files with columns:
   - `article` or `clean_text`: Article text
   - `highlights` or `Summary`: Reference summaries

2. Upload the dataset to Google Drive:
   - Create a folder in Google Drive (e.g., `processed/10k_samples/`)
   - Upload `train.csv`, `val.csv`, and `test.csv`

### 2. Open Notebook in Colab

1. Open the appropriate notebook in Google Colab
2. Make sure to select a GPU runtime: **Runtime → Change runtime type → GPU**

### 3. Configure the Notebook

1. Update `DRIVE_DATA_PATH` in the first cell to point to your dataset:
   ```python
   DRIVE_DATA_PATH = "/content/drive/MyDrive/processed/10k_samples"
   ```

2. Adjust training parameters if needed:
   - `EPOCHS`: Number of training epochs
   - `BATCH_SIZE`: Batch size (adjust based on GPU memory)
   - `MAX_LENGTH`: Maximum input length in tokens

### 4. Run All Cells

Execute all cells sequentially. The notebook will:
- Mount Google Drive
- Install required packages
- Load and preprocess your dataset
- Fine-tune the model
- Evaluate on validation set
- Save the fine-tuned model

## 📊 Model Options

### T5 Models

- **T5 Base**: Faster training, smaller model size (~220M parameters)
- **T5 Large**: Better quality, larger model size (~770M parameters)
- **FLAN-T5**: Instruction-tuned variant, often better performance

**Notebooks**: `t5_base/` or `t5_large/`

### Pegasus Models

- **Pegasus XSum**: Trained on XSum dataset (news summarization)
- **Pegasus CNN/DailyMail**: Trained on CNN/DailyMail dataset
- **Pegasus Large**: Larger variant with better quality

**Notebooks**: `pegasus/` or `daily_mail/`

## ⚙️ Configuration Guide

### Key Parameters

```python
MODEL = 'google/pegasus-cnn_dailymail'  # Model to fine-tune
EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 4  # Training batch size (adjust for GPU memory)
MAX_LENGTH = 1024  # Maximum input length in tokens
OUT_DIR = 'results-pegasus/2k_samples'  # Output directory
```

### Batch Size Guidelines

- **T5 Base**: Start with batch size 8-16
- **T5 Large**: Start with batch size 2-4
- **Pegasus**: Start with batch size 4-8

If you get out-of-memory errors, reduce `BATCH_SIZE` or `MAX_LENGTH`.

### Learning Rate

Default learning rates are set in the notebooks. Adjust if needed:
- T5: Typically 1e-4 to 5e-4
- Pegasus: Typically 3e-5 to 1e-4

## 📈 Monitoring Training

### TensorBoard

The notebooks automatically log metrics to TensorBoard. To view:

1. In Colab, run:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir {OUT_DIR}
   ```

2. Or download logs and view locally:
   ```bash
   tensorboard --logdir {OUT_DIR}
   ```

### Metrics Tracked

- Training loss
- Validation loss
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- GPU memory usage
- Learning rate schedule

## 💾 Saving Models

### Save to Google Drive

After training, models are saved to the `OUT_DIR`. To save to Google Drive:

```python
!cp -r {OUT_DIR} /content/drive/MyDrive/models/
```

### Save to Google Cloud Storage

If you've set up GCS (see notebook setup cells):

```python
!gsutil cp -r {OUT_DIR} gs://your-bucket-name/models/
```

## 🔍 Evaluation

The notebooks automatically evaluate models on the validation set using:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence

Results are saved to CSV files in the output directory.

## 🐛 Troubleshooting

### Out of Memory Errors

1. Reduce `BATCH_SIZE`
2. Reduce `MAX_LENGTH`
3. Enable gradient accumulation (already configured in notebooks)
4. Use a smaller model (t5-base instead of t5-large)

### Slow Training

1. Ensure GPU is enabled: **Runtime → Change runtime type → GPU**
2. Increase `BATCH_SIZE` if memory allows
3. Reduce `MAX_LENGTH` if articles are very long
4. Use mixed precision training (bf16/fp16) - already enabled

### Dataset Not Found

1. Verify `DRIVE_DATA_PATH` is correct
2. Ensure Google Drive is mounted
3. Check that CSV files exist at the specified path

## 📝 Notes

- Training time varies based on dataset size, model size, and GPU type
- T5 Large and Pegasus Large require more GPU memory
- Early stopping is enabled to prevent overfitting
- Best model (lowest validation loss) is automatically loaded after training

## 🔗 Related Files

- Local training script: `scripts/train_summarizers.py`
- Model usage: `scripts/summarizers/`
- Main pipeline: `pipeline_with_training.py`

---

**Tip**: Start with a smaller dataset (2k samples) to test the setup, then scale up to larger datasets (10k, 50k) once everything works.

