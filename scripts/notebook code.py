
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from tqdm import tqdm
import json
from pathlib import Path
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torch.backends.cudnn.conv.fp32_precision = 'tf32'
# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_length=512, max_target_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        article = str(row["clean_text"])
        summary = str(row["Summary"])

        model_inputs = self.tokenizer(
            article,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            text_target=summary,
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        # squeeze batch dimension
        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"][0]
        labels = labels.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Load data from Google Drive
DRIVE_DATA_PATH = '/content/drive/MyDrive/processed/test/'  # Update this path
print("Loading data from Google Drive...")
train_df = pd.read_csv(f"{DRIVE_DATA_PATH}/train.csv").head(10)
val_df = pd.read_csv(f"{DRIVE_DATA_PATH}/val.csv").head(5)
test_df = pd.read_csv(f"{DRIVE_DATA_PATH}/test.csv").head(5)

# Filter out rows with missing summaries
train_df = train_df.dropna(subset=['Summary', 'clean_text'])
val_df = val_df.dropna(subset=['Summary', 'clean_text'])

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Initialize T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.enable_input_require_grads()
model.train()

print(f"Model loaded on: {next(model.parameters()).device}")

# Create datasets
train_dataset = SummarizationDataset(train_df, tokenizer)
val_dataset = SummarizationDataset(val_df, tokenizer)
print(train_dataset)
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


# Training arguments
output_dir = '/content/t5-finetuned'
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,        # you can increase!
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,        # effective batch size = 16
    learning_rate=1e-5,
    warmup_steps=200,
    weight_decay=0.01,

    fp16=False,   # A100: No need for fp16
    bf16=True,    # A100: best precision for training
    tf32=True,    # A100: enables faster matmul
    max_grad_norm=1.0,

    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_checkpointing=False,

    dataloader_num_workers=4,
    report_to="none",
)

model.config.task_specific_params = {
    "summarization": {
        "early_stopping": True,
        "length_penalty": 2.0,
        "max_length": 200,
        "min_length": 30,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "prefix": "summarize: "
    }
}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train
print(f"Training on {len(train_df)} samples...")
trainer.train()

# Save model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")


# Load fine-tuned model for inference
model.eval()
results = []

# Process test set (limit to first 50 for demo)
test_limit = 50
test_subset = test_df.head(test_limit)

print(f"Processing {len(test_subset)} articles...")

for idx, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
    article_id = int(idx)
    text = str(row["clean_text"])

    # Generate summary
    prompt = "summarize: " + text
    encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.inference_mode():
        output = model.generate(
            **encoding,
            max_length=128,
            num_beams=1,
            do_sample=False
        )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "article_id": article_id,
        "original_text": text,
        "summary": summary
    })

print(f"Processed {len(results)} articles")
