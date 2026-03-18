# 📝 Text Summarizer

An AI-powered text summarization tool that uses both **extractive** and **abstractive** methods to generate concise summaries from long documents.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Dual Summarization Methods**:
  - **Extractive**: Uses TextRank algorithm to identify and extract key sentences
  - **Abstractive**: Uses transformer models (BART, T5) to generate new summary text

- **Multiple Input Options**:
  - Direct text input
  - File upload support (PDF, TXT, DOCX)

- **Customizable Output**:
  - Adjustable summary length
  - Multiple AI model options
  - Side-by-side comparison of methods

- **User-Friendly Interface**:
  - Modern Streamlit web interface
  - Real-time statistics
  - Download summaries as text files

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd "Major Project"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (first-time setup):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('punkt_tab')
   ```

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Summarizer

1. **Select Summarization Method**:
   - Choose between Extractive, Abstractive, or Both (for comparison)

2. **Configure Settings**:
   - For Extractive: Set the number of sentences to extract
   - For Abstractive: Choose the AI model and summary length

3. **Input Text**:
   - **Tab 1**: Paste or type text directly
   - **Tab 2**: Upload a PDF, TXT, or DOCX file

4. **Generate Summary**:
   - Click "Summarize" to generate the summary
   - View statistics and download the result

### Programmatic Usage

You can also use the summarizers directly in Python:

```python
# Extractive Summarization
from summarizers import ExtractiveSummarizer

summarizer = ExtractiveSummarizer()
result = summarizer.summarize(text, num_sentences=3)
print(result["summary"])

# Abstractive Summarization
from summarizers import AbstractiveSummarizer

summarizer = AbstractiveSummarizer(model_name="distilbart")
result = summarizer.summarize(text, max_length=150)
print(result["summary"])
```

## 📁 Project Structure

```
Major Project/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── summarizers/           # Summarization modules
│   ├── __init__.py
│   ├── extractive.py      # TextRank-based extraction
│   └── abstractive.py     # Transformer-based generation
└── utils/                 # Utility modules
    ├── __init__.py
    └── file_handler.py    # File processing utilities
```

## 🤖 Available Models

| Model | Description | Speed | Quality |
|-------|-------------|-------|---------|
| `distilbart` | DistilBART CNN (default) | ⚡ Fast | Good |
| `bart` | BART Large CNN | 🐢 Slow | Excellent |
| `t5` | T5 Small | ⚡ Fast | Good |
| `t5-base` | T5 Base | 🐢 Medium | Very Good |

## 📊 How It Works

### Extractive Summarization (TextRank)

1. **Sentence Tokenization**: Split text into individual sentences
2. **Similarity Matrix**: Calculate cosine similarity between all sentence pairs
3. **PageRank Algorithm**: Apply graph-based ranking to identify important sentences
4. **Selection**: Extract top-ranked sentences maintaining original order

### Abstractive Summarization (Transformers)

1. **Tokenization**: Convert text to model-compatible tokens
2. **Encoding**: Process input through transformer encoder
3. **Generation**: Use beam search to generate summary tokens
4. **Decoding**: Convert generated tokens back to human-readable text

## ⚠️ Notes

- First-time usage will download AI models (~500MB-2GB)
- Abstractive summarization requires more computation
- GPU acceleration available if CUDA-compatible GPU is present
- Large documents are automatically chunked for processing

## 🔧 Troubleshooting

**Model download issues:**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/
```

**NLTK data issues:**
```python
import nltk
nltk.download('all')
```

**Memory issues with large files:**
- Use extractive summarization for very long documents
- Consider using a smaller model like `t5-small`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK](https://www.nltk.org/)
- [Streamlit](https://streamlit.io/)
- [NetworkX](https://networkx.org/)

---

**Developed for Major Project - 2026**
