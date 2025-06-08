# Distributed Training with Hugging Face Accelerate

This module demonstrates how to implement distributed training for transformer models using Hugging Face Accelerate. It provides a complete pipeline for fine-tuning pre-trained models on classification tasks across multiple GPUs or machines.

## 🎯 Learning Objectives

After working through this module, you will understand:
- How to set up distributed training with Hugging Face Accelerate
- The differences between single-GPU and multi-GPU training
- How to handle data loading and processing for distributed training
- Best practices for model evaluation in distributed settings
- How to save and share trained models

## 📁 Project Structure

```
Distributed_training/
├── src/
│   ├── data_connector.py       # Data loading utilities
│   ├── data_processing.py      # Text tokenization and preprocessing
│   ├── model.py               # Model and tokenizer setup
│   ├── training.py            # Distributed training implementation
│   ├── training_application.py # Main application script
│   └── requirements.txt       # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd code-examples/LLMs/Distributed_training
pip install -r src/requirements.txt
```

### 2. Configure Accelerate (First Time Only)

```bash
accelerate config
```

Follow the prompts to configure your training environment:
- **Single GPU**: Select "No distributed training"
- **Multi-GPU**: Select "multi-GPU" and specify number of GPUs
- **Multi-machine**: Select "multi-GPU" and configure machine settings

### 3. Run Training

**Single GPU:**
```bash
cd src
python training_application.py
```

**Multiple GPUs:**
```bash
cd src
accelerate launch --multi_gpu training_application.py
```

**Multiple Machines:**
```bash
cd src
accelerate launch --multi_gpu --num_machines 2 training_application.py
```

## 📚 Code Overview

### Core Components

#### 1. DataConnector (`data_connector.py`)
- Simple interface for loading datasets from Hugging Face Hub
- Abstracts dataset loading logic for easy swapping of datasets

#### 2. DataProcessor (`data_processing.py`)
- Handles text tokenization and preprocessing
- Converts raw text to model-ready format
- Manages padding, truncation, and format conversion

#### 3. Model Setup (`model.py`)
- Loads pre-trained models and tokenizers
- Configures models for classification tasks
- Handles padding token setup for models like GPT-2

#### 4. Distributed Trainer (`training.py`)
- Implements distributed training using Accelerate
- Handles multi-GPU coordination automatically
- Includes evaluation and model saving functionality

#### 5. Main Application (`training_application.py`)
- Orchestrates the complete training pipeline
- Demonstrates end-to-end usage of all components
- Easy to modify for different datasets and models

### Key Features

- **Automatic Distributed Setup**: Accelerate handles device placement and distributed coordination
- **Flexible Model Support**: Works with any Hugging Face transformer model
- **Efficient Data Loading**: Batched processing and proper data loader setup
- **Comprehensive Evaluation**: Distributed evaluation with metric aggregation
- **Model Sharing**: Automatic upload to Hugging Face Hub

## 🔧 Configuration

### Modifying the Training Setup

Edit the configuration variables in `training_application.py`:

```python
MODEL_ID = 'gpt2'  # Change to your preferred model
DATA_PATH = 'dair-ai/emotion'  # Change to your dataset
```

### Training Parameters

Adjust training parameters in the `Trainer` initialization:

```python
trainer = Trainer(
    model=model, 
    tokenizer=tokenizer, 
    num_epochs=3,      # Number of training epochs
    batch_size=16      # Batch size per device
)
```

### Hugging Face Hub Upload

To enable model upload, replace the token in `training.py`:

```python
HUGGINGFACE_TOKEN = "your_actual_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

## 🔍 Understanding Distributed Training

### What is Distributed Training?

Distributed training allows you to:
- **Scale training** across multiple GPUs or machines
- **Reduce training time** by parallelizing computations
- **Train larger models** that don't fit on a single GPU
- **Process larger datasets** more efficiently

### How Accelerate Works

Accelerate simplifies distributed training by:
1. **Automatic device detection** and placement
2. **Gradient synchronization** across devices
3. **Data loading coordination** to avoid duplication
4. **Loss and metric aggregation** for accurate evaluation

### Key Concepts

- **Data Parallel Training**: Each device processes a different batch of data
- **Gradient Accumulation**: Gradients are averaged across all devices
- **Evaluation Gathering**: Predictions are collected from all devices for metrics

## 📊 Example Datasets

The code works with any Hugging Face classification dataset. Here are some examples:

- `dair-ai/emotion` - Emotion classification (6 classes)
- `imdb` - Movie review sentiment (2 classes)
- `ag_news` - News categorization (4 classes)
- `yelp_polarity` - Restaurant review sentiment (2 classes)

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size in `training_application.py`
- Use gradient accumulation: `accelerate config` → enable gradient accumulation

**Slow Training:**
- Increase batch size if you have more GPU memory
- Enable mixed precision in `accelerate config`
- Use faster data loading with more workers

**Model Upload Fails:**
- Check your Hugging Face token
- Ensure you have write permissions
- Verify internet connection

### Performance Tips

1. **Batch Size**: Larger batches generally train faster but use more memory
2. **Mixed Precision**: Enable FP16 for faster training with minimal accuracy loss
3. **Data Loading**: Use multiple workers for faster data loading
4. **Gradient Accumulation**: Simulate larger batches when memory is limited

## 🔬 Experimentation Ideas

1. **Try Different Models**: 
   - Compare BERT vs GPT-2 vs RoBERTa
   - Test model sizes (base vs large)

2. **Dataset Experiments**:
   - Train on different classification tasks
   - Compare performance across domains

3. **Training Optimizations**:
   - Experiment with learning rates
   - Try different optimizers (AdamW, SGD)
   - Test various batch sizes

4. **Distributed Scaling**:
   - Measure training time vs number of GPUs
   - Compare single vs multi-machine setups

## 📈 Expected Results

With the default configuration (GPT-2 on emotion dataset):
- **Training Time**: ~10-15 minutes on single GPU
- **Expected Accuracy**: ~85-90% on validation set
- **Memory Usage**: ~2-3GB GPU memory

## 🤝 Contributing

This is an educational module. Feel free to:
- Add support for different model types
- Implement additional evaluation metrics
- Create examples with larger datasets
- Add visualization tools for training progress

## 📖 Additional Resources

- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Distributed Training Guide](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Model Hub](https://huggingface.co/models)