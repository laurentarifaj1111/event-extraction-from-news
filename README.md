# Event Extraction from News Articles

A comprehensive NLP pipeline for extracting events and generating summaries from news articles using state-of-the-art transformer models. This project implements multiple summarization approaches (extractive and abstractive) and provides tools for fine-tuning models on custom datasets.

## 📋 Project Overview

This project implements an end-to-end natural language processing pipeline that:

1. **Extracts Events** from news articles using Named Entity Recognition (NER) and event classification
   - Identifies entities (persons, organizations, locations)
   - Classifies event types (conflict, politics, legal, economy, disaster)
   - Detects event trigger words

2. **Generates Summaries** using multiple summarization approaches:
   - **BERT Extractive Summarization**: Selects and combines important sentences from the original text
   - **Pegasus Abstractive Summarization**: Generates new summary text using a pre-trained abstractive model
   - **T5 Abstractive Summarization**: Uses the Text-to-Text Transfer Transformer for abstractive summarization

3. **Evaluates Performance** using ROUGE scores and BERTScore metrics
   - ROUGE-1, ROUGE-2, ROUGE-L for measuring summary quality
   - BERTScore for semantic similarity (optional)
   - Inference time measurement

4. **Supports Fine-tuning** of models on custom datasets for improved performance
   - Local fine-tuning via Python scripts
   - Google Colab notebooks for GPU-accelerated training
   - Automatic model checkpointing and best model selection

## 🏗️ Project Structure

```
event-extraction-from-news/
├── colab_notebook/          # Google Colab notebooks for model fine-tuning
│   ├── t5_base/            # T5 base model fine-tuning notebooks
│   ├── t5_large/           # T5 large model fine-tuning notebooks
│   ├── pegasus/            # Pegasus model fine-tuning notebooks
│   └── daily_mail/         # CNN/DailyMail dataset experiments
├── scripts/                 # Core Python scripts
│   ├── preprocessing.py    # Data cleaning and preprocessing
│   ├── event_extraction.py # Event and entity extraction
│   ├── evaluation.py       # Model evaluation metrics
│   ├── train_summarizers.py # Model fine-tuning script
│   ├── create_small_train.py # Dataset creation utilities
│   ├── filter_by_length.py  # Dataset filtering by length
│   └── summarizers/         # Summarization model implementations
│       ├── summarizer_bert.py
│       ├── summarizer_pegasus.py
│       └── summarizer_t5.py
├── data/                    # Data directory
│   ├── raw/                 # Raw dataset files
│   ├── processed/           # Preprocessed train/val/test splits
│   └── filtered_v1/         # Filtered datasets
├── notebooks/               # Local Jupyter notebooks for analysis
├── results/                 # Model outputs and evaluation results
├── pipeline.py              # Main pipeline (pretrained models)
└── pipeline_with_training.py # Pipeline with fine-tuned model support
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS support)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd event-extraction-from-news
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if not already downloaded):
```python
import nltk
nltk.download('punkt')
```

### Basic Usage

#### Run the Pipeline with Pretrained Models

```bash
python pipeline.py --limit 50
```

This will:
- Load and preprocess the dataset
- Extract events from articles
- Generate summaries using BERT, Pegasus, and T5
- Evaluate summaries using ROUGE and BERTScore
- Save results to `results/structured_results.json`

#### Run with Fine-tuned Models

```bash
python pipeline_with_training.py --limit 50
```

This pipeline automatically detects and uses fine-tuned models if available, otherwise falls back to pretrained models.

## 📊 Models

### Event Extraction

- **Model**: `dslim/bert-base-NER`
- **Task**: Named Entity Recognition (NER) and event classification
- **Output**: Extracted entities (persons, organizations, locations) and event types (conflict, politics, legal, economy, disaster)

### Summarization Models

#### 1. BERT Extractive Summarizer
- **Type**: Extractive (selects sentences from original text)
- **Model**: `bert-base-uncased`
- **Use Case**: Fast, reliable summaries that preserve original wording
- **Fine-tuning**: Not typically required (uses sentence ranking)

#### 2. Pegasus Summarizer
- **Type**: Abstractive (generates new text)
- **Model**: `google/pegasus-xsum` or `google/pegasus-cnn_dailymail`
- **Use Case**: High-quality abstractive summaries
- **Fine-tuning**: Supported via `scripts/train_summarizers.py` or Colab notebooks

#### 3. T5 Summarizer
- **Type**: Abstractive (generates new text)
- **Model**: `google/flan-t5-base` or `t5-large`
- **Use Case**: Flexible text-to-text generation
- **Fine-tuning**: Supported via `scripts/train_summarizers.py` or Colab notebooks

## 🔧 Fine-tuning Models

### Local Fine-tuning

Use the provided training script:

```bash
python scripts/train_summarizers.py
```

This will:
- Load training and validation data from `data/processed/`
- Fine-tune Pegasus and T5 models
- Save fine-tuned models to `models/pegasus-finetuned/` and `models/t5-finetuned/`

### Google Colab Fine-tuning

For GPU-accelerated training, use the notebooks in `colab_notebook/`:

1. **T5 Models**: Use notebooks in `colab_notebook/t5_base/` or `colab_notebook/t5_large/`
2. **Pegasus Models**: Use notebooks in `colab_notebook/pegasus/`
3. **CNN/DailyMail**: Use notebooks in `colab_notebook/daily_mail/`

#### Colab Setup Steps:

1. Upload your dataset to Google Drive
2. Open the appropriate notebook in Google Colab
3. Update `DRIVE_DATA_PATH` to point to your dataset
4. Run all cells to fine-tune the model
5. Download the fine-tuned model or save to Google Cloud Storage

## 📈 Evaluation Metrics

The pipeline evaluates summaries using:

- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence (captures sentence structure)
- **BERTScore**: Semantic similarity using BERT embeddings (optional, slower)

## 🛠️ Scripts Reference

### Data Processing

- **`scripts/preprocessing.py`**: Data cleaning, text normalization, train/val/test splitting
- **`scripts/create_small_train.py`**: Create smaller training datasets from raw data
  ```bash
  python scripts/create_small_train.py --num 2000 --output data/processed
  ```
- **`scripts/filter_by_length.py`**: Filter articles by word length and quality metrics
  ```bash
  python scripts/filter_by_length.py --max-words 500 --sample-percent 0.1
  ```

### Model Training

- **`scripts/train_summarizers.py`**: Fine-tune Pegasus and T5 models locally
- **`colab_notebook/*/`: Colab notebooks for GPU-accelerated training

### Model Usage

- **`scripts/event_extraction.py`**: Extract events and entities from text
- **`scripts/summarizers/summarizer_bert.py`**: BERT extractive summarization
- **`scripts/summarizers/summarizer_pegasus.py`**: Pegasus abstractive summarization
- **`scripts/summarizers/summarizer_t5.py`**: T5 abstractive summarization

### Evaluation

- **`scripts/evaluation.py`**: Compute ROUGE and BERTScore metrics

## 📁 Data Format

### Input Data

The pipeline expects CSV files with the following columns:
- `Content` or `article`: The article text
- `Summary` or `highlights`: Reference summary (for training/evaluation)

### Output Format

Results are saved as JSON with the following structure:

```json
{
  "article_id": 0,
  "original_text": "...",
  "event_extraction": {
    "entities": [...],
    "event_type": "...",
    "event_trigger": "..."
  },
  "summaries": {
    "bert": "...",
    "pegasus": "...",
    "t5": "..."
  },
  "metrics": {
    "bert": {
      "rouge1": 0.xx,
      "rouge2": 0.xx,
      "rougeL": 0.xx,
      "time_ms": xxx
    },
    ...
  }
}
```

## 🎯 Use Cases

1. **News Summarization**: Automatically summarize news articles
2. **Event Monitoring**: Extract and classify events from news streams
3. **Content Analysis**: Analyze and summarize large volumes of text
4. **Research**: Compare different summarization approaches

## 🔍 Performance Optimization

### GPU Acceleration

The project supports:
- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2/M3) GPUs
- **CPU**: Fallback for systems without GPU

Device selection is automatic via `scripts/utils.py`.

### Performance Tips

1. **Batch Processing**: Process multiple articles in batches
2. **Model Quantization**: Use smaller models (t5-base vs t5-large) for faster inference
3. **Skip BERTScore**: BERTScore is computationally expensive; skip it for faster evaluation
4. **Fine-tune Models**: Fine-tuned models often perform better on domain-specific data

## 📝 Configuration

### Model Configuration

Edit model parameters in the respective summarizer files:
- `scripts/summarizers/summarizer_bert.py`: Adjust `min_length`, `max_length`
- `scripts/summarizers/summarizer_pegasus.py`: Adjust `max_len`
- `scripts/summarizers/summarizer_t5.py`: Adjust `max_len`

### Training Configuration

For local training, edit `scripts/train_summarizers.py`:
- Batch sizes
- Learning rates
- Number of epochs
- Early stopping patience

For Colab training, edit the configuration cell in the notebook:
- `MODEL`: Model name
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Training batch size
- `MAX_LENGTH`: Maximum input length

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- HuggingFace Transformers for pre-trained models
- Google Research for T5 and Pegasus models
- The open-source NLP community

## 📧 Contact

[Add your contact information here]

---

**Note**: This project is designed for research and educational purposes. Ensure you have appropriate licenses for any datasets you use.

