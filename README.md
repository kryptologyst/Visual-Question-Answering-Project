# Visual Question Answering Project

A production-ready implementation of Visual Question Answering (VQA) using state-of-the-art transformer models from Hugging Face.

## Features

- **Multiple Model Support**: BLIP, BLIP-2, and other VQA models
- **Web Interface**: Streamlit-based interactive demo
- **CLI Interface**: Command-line tool for batch processing
- **Modern Architecture**: Type hints, error handling, logging, and configuration management
- **Extensible Design**: Easy to add new models and features
- **Comprehensive Testing**: Unit tests and integration tests
- **Production Ready**: Proper error handling, logging, and documentation

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Streamlit (for web interface)
- PIL/Pillow
- NumPy
- Matplotlib

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Visual-Question-Answering-Project.git
cd Visual-Question-Answering-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501` and start asking questions about images!

### Command Line Interface

Basic usage:
```bash
python cli.py --image path/to/image.jpg --question "What is in this image?"
```

Interactive mode:
```bash
python cli.py --interactive
```

Batch processing:
```bash
python cli.py --batch images/ questions.txt --output results.json
```

### Python API

```python
from src.vqa_model import VQAModel

# Initialize model
model = VQAModel(model_name="Salesforce/blip-vqa-base")

# Answer a question
result = model.answer_question("path/to/image.jpg", "What do you see?")
print(f"Answer: {result.answer}")
```

## 📁 Project Structure

```
visual-question-answering/
├── src/                    # Source code
│   ├── vqa_model.py       # Core VQA implementation
│   └── config.py          # Configuration management
├── web_app/               # Streamlit web interface
│   └── app.py
├── tests/                 # Test suite
│   └── test_vqa.py
├── config/                # Configuration files
│   └── config.yaml
├── data/                  # Data directory (created automatically)
├── models/                # Model cache directory
├── outputs/               # Output directory
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 🔧 Configuration

The project uses YAML configuration files. You can modify `config/config.yaml` to customize:

- Model selection and parameters
- Application settings
- UI preferences

Example configuration:
```yaml
model:
  model_name: "Salesforce/blip-vqa-base"
  device: "auto"  # or "cpu", "cuda", "mps"
  max_length: 50
  num_beams: 5
  temperature: 1.0

app:
  debug: false
  log_level: "INFO"

ui:
  title: "Visual Question Answering Demo"
  theme: "light"
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or run specific tests:
```bash
python tests/test_vqa.py
```

## Supported Models

- **BLIP Base**: `Salesforce/blip-vqa-base` (default)
- **BLIP Large**: `Salesforce/blip-vqa-capfilt-large`
- **BLIP-2**: `Salesforce/blip2-opt-2.7b`

## Example Questions

Try asking these types of questions:

- "What is in this image?"
- "What colors do you see?"
- "How many objects are there?"
- "What is the main subject?"
- "Describe the scene"
- "What is the mood or atmosphere?"
- "Is it daytime or nighttime?"
- "What is the person doing?"

## Advanced Usage

### Custom Model Loading

```python
from src.vqa_model import VQAModel

# Load a specific model
model = VQAModel(
    model_name="Salesforce/blip-vqa-capfilt-large",
    device="cuda",
    use_pipeline=True
)
```

### Batch Processing

```python
# Process multiple images
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
questions = ["What is this?", "What colors?", "How many objects?"]

results = model.batch_answer(images, questions)
for result in results:
    print(f"Q: {result.question}")
    print(f"A: {result.answer}")
```

### Configuration Management

```python
from src.config import ConfigManager

config_manager = ConfigManager("config/custom_config.yaml")
model_config = config_manager.get_model_config()
app_config = config_manager.get_app_config()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use CPU or reduce batch size
2. **Model Loading Errors**: Check internet connection and model availability
3. **Image Format Issues**: Ensure images are in supported formats (JPG, PNG)

### Performance Tips

- Use GPU acceleration when available
- Reduce `max_length` for faster generation
- Use `num_beams=1` for faster (but less accurate) results
- Enable `use_pipeline=True` for simpler usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Salesforce](https://www.salesforce.com/) for the BLIP models
- [Streamlit](https://streamlit.io/) for the web interface framework

## References

- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)


# Visual-Question-Answering-Project
