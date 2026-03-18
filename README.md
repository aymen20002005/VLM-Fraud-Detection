# VLM Fraud Detection

A multimodal fraud detection system that uses Google's Gemini 2.5 Pro vision-language model to analyze PDF documents for fraudulent content.

## Overview

This project processes PDF documents and uses Gemini's advanced vision capabilities to detect potential fraud indicators such as:
- Visual inconsistencies and tampering
- Altered fields
- Unrealistic values
- Suspicious patterns

The system provides:
- **Fraud probability** (0-1 scale)
- **Fraud classification** (fraud or legitimate)
- **Suspicious elements** list
- **Visual observations** from the document

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd VLM-Fraud-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` and add your Google Cloud Project ID:
```
PROJECT_ID=your-gcp-project-id
LOCATION=us-central1
MODEL_NAME=gemini-2.5-pro
TEMPERATURE=0.2
MAX_OUTPUT_TOKENS=1024
```

## Usage

Process a PDF document:
```bash
python main.py path/to/document.pdf
```

### Output Example

```
=== FRAUD DETECTION RESULT ===
fraud_probability: 0.15
fraud_label: legitimate
suspicious_elements: ['Signature alignment appears off']
visual_observations: ['Document quality is good', 'All fields are clearly printed']
explanation: The document shows minor inconsistencies but overall appears legitimate...
```

## Project Structure

```
.
├── main.py                 # Entry point
├── fraud_detector.py       # Core fraud detection logic
├── config.py              # Configuration settings
├── prompts.py             # LLM prompt templates
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (local, not in git)
├── .env.example          # Environment variables template
└── dataset/              # Sample datasets
```

## How It Works

1. **PDF Loading**: The system reads the PDF file directly in binary format
2. **Model Processing**: Sends the PDF to Gemini 2.5 Pro as a multimodal input
3. **Analysis**: The model analyzes the visual content for fraud indicators
4. **JSON Response**: Returns structured fraud assessment

## License

[Add your license here]

## References

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Documentation](https://ai.google.dev/)
