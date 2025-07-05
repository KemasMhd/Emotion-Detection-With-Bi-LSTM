# Setup Guide

This guide will help you set up the Emotion Detection project on your local machine.

## Prerequisites

- Python 3.8 or higher
- Git
- Jupyter Notebook or JupyterLab

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Emotion_Detection_With_Bi_LSTM.git
cd Emotion_Detection_With_Bi_LSTM
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```python
python -c "
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"
```

### 5. Verify Installation

```bash
python -c "
import pandas as pd
import sklearn
import matplotlib
import seaborn
import nltk
import transformers
print('✅ All packages installed successfully!')
"
```

### 6. Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

### 7. Run the Analysis

1. Navigate to `notebooks/Emotion_Detection_With_Bi_LSTM.ipynb`
2. Run all cells sequentially
3. View results and visualizations

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure virtual environment is activated and all packages are installed
2. **NLTK Data Error**: Run the NLTK download commands in step 4
3. **Permission Error**: Run with administrator privileges if needed
4. **Memory Issues**: Close other applications if running out of memory during model training

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: ~1GB free space
- **Internet**: Required for downloading models and data

## Next Steps

After successful setup:

1. Explore the main notebook
2. Examine data in the `data/` directory
3. Check results and visualizations
4. Modify code for your specific use case

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed error information
