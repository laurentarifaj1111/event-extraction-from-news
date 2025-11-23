"""
Fine-tuning script for summarization models.
Trains Pegasus and T5 on train.csv data.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    PegasusTokenizer, PegasusForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from scripts.utils import get_device, print_device_info

DEVICE = get_device()


class SummarizationDataset(Dataset):
    """Dataset for summarization fine-tuning."""
    
    def __init__(self, dataframe, tokenizer, max_input_length=512, max_target_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Use clean_text as input, Summary as target
        article = str(row['clean_text'])
        summary = str(row['Summary'])
        
        # Tokenize inputs
        inputs = self.tokenizer(
            article,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }


def train_pegasus(train_df, val_df, output_dir='models/pegasus-finetuned'):
    """Fine-tune Pegasus model."""
    print("\n" + "="*50)
    print("Fine-tuning PEGASUS")
    print("="*50)
    print(f"Using device: {DEVICE}")
    
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    print(f"Model loaded on: {next(model.parameters()).device}")
    
    # Create datasets
    train_dataset = SummarizationDataset(train_df, tokenizer)
    val_dataset = SummarizationDataset(val_df, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Increased from 2 for better GPU utilization
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # Effective batch size = 4 * 2 = 8
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy="epoch",  # Changed from eval_strategy for compatibility
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=False,  # Set to True if using CUDA
        dataloader_num_workers=4,  # Parallel data loading to reduce CPU bottleneck
        dataloader_pin_memory=True,  # Faster CPU->GPU transfer
    )
    
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
    
    return model, tokenizer


def train_t5(train_df, val_df, output_dir='models/t5-finetuned'):
    """Fine-tune T5 model."""
    print("\n" + "="*50)
    print("Fine-tuning T5")
    print("="*50)
    print(f"Using device: {DEVICE}")
    
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    print(f"Model loaded on: {next(model.parameters()).device}")
    
    # Create datasets
    train_dataset = SummarizationDataset(train_df, tokenizer)
    val_dataset = SummarizationDataset(val_df, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Increased from 2 for better GPU utilization
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # Effective batch size = 4 * 2 = 8
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy="epoch",  # Changed from eval_strategy for compatibility
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=False,  # Set to True if using CUDA
        dataloader_num_workers=4,  # Parallel data loading to reduce CPU bottleneck
        dataloader_pin_memory=True,  # Faster CPU->GPU transfer
    )
    
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
    
    return model, tokenizer


def main():
    """Main training function."""
    # Print device information
    print_device_info()
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv("../data/processed/train.csv")
    val_df = pd.read_csv("../data/processed/val.csv")
    
    # Filter out rows with missing summaries
    train_df = train_df.dropna(subset=['Summary', 'clean_text'])
    val_df = val_df.dropna(subset=['Summary', 'clean_text'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    
    # Fine-tune models
    print("\n" + "="*50)
    print("Starting Fine-tuning")
    print("="*50)
    
    # Train Pegasus
    train_pegasus(train_df, val_df)
    
    # Train T5
    train_t5(train_df, val_df)
    
    print("\n" + "="*50)
    print("Fine-tuning Complete!")
    print("="*50)


if __name__ == "__main__":
    main()

