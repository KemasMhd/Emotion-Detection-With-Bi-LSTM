# 🧠 Emotion Detection with Machine Learning on Counseling Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Science](https://img.shields.io/badge/Data-Science-purple.svg)](notebooks/)

## 📋 Project Overview

This project implements an advanced **emotion detection system** using machine learning to analyze and classify emotions in counseling conversation texts. The system processes data from the CounselChat dataset, which contains real counseling questions and responses, to automatically detect emotional states that can assist mental health professionals in understanding client needs.

### 🎯 Key Features

- **📊 Comprehensive EDA**: Rich data exploration with visualizations and statistical analysis
- **🧹 Advanced Text Preprocessing**: Text cleaning, tokenization, stopword removal, and lemmatization
- **🤖 Automated Emotion Labeling**: Uses Cardiff NLP's RoBERTa model for emotion annotation
- **🎯 Multiple ML Models**: Logistic Regression, SVM, Random Forest, Naive Bayes comparison
- **🎭 Ensemble Methods**: Voting classifier for improved accuracy
- **📈 Professional Visualization**: Word clouds, confusion matrices, performance metrics
- **📝 Complete Documentation**: Detailed analysis workflow and insights

## 📊 Dataset Information

**Source**: CounselChat Dataset

- **Total Records**: 1,482 counseling questions
- **Features**: 9 columns including question text, topics, therapist responses
- **Text Statistics**: Average 54 words per question (275 characters)
- **Missing Data**: Minimal (< 7% in any column)

### 🏷️ Top Counseling Topics

| Rank | Topic           | Questions | Percentage |
| ---- | --------------- | --------- | ---------- |
| 1    | Relationships   | 383       | 25.8%      |
| 2    | Anxiety         | 234       | 15.8%      |
| 3    | Depression      | 205       | 13.8%      |
| 4    | Intimacy        | 198       | 13.4%      |
| 5    | Family Conflict | 182       | 12.3%      |

### 😊 Emotion Distribution

| Emotion      | Count | Percentage |
| ------------ | ----- | ---------- |
| **Sadness**  | 1,072 | 72.3%      |
| **Joy**      | 250   | 16.9%      |
| **Optimism** | 152   | 10.3%      |
| **Anger**    | 8     | 0.5%       |

## 🏗️ Project Structure

```
Emotion_Detection_With_Bi_LSTM/
├── 📁 data/                    # Datasets (original & processed)
│   ├── counselchat-data.csv    # Original counseling dataset
│   ├── labeled_emotion_data.csv # Processed data with emotions
│   └── README.md               # Data documentation
├── 📁 notebooks/               # Jupyter notebooks
│   ├── Emotion_Detection_With_Bi_LSTM.ipynb  # Main analysis
│   └── README.md               # Notebook documentation
├── 📁 src/                     # Source code (future development)
│   └── README.md               # Code structure guide
├── 📁 results/                 # Model outputs & metrics
│   └── README.md               # Results documentation
├── 📁 docs/                    # Additional documentation
│   └── README.md               # Documentation guide
├── 📄 README.md                # This file
├── 📄 requirements.txt         # Python dependencies
├── 📄 LICENSE                  # MIT License
└── 📄 .gitignore              # Git ignore rules
```

## 🔧 Technical Implementation

### Data Processing Pipeline

```
Raw Text → Cleaning → Stopword Removal → Lemmatization → TF-IDF → ML Models
```

### Model Architecture

- **Feature Engineering**: TF-IDF Vectorization (5,000 features, unigrams + bigrams)
- **Preprocessing**: ~44% text size reduction while preserving meaning
- **Class Balancing**: Weighted classes to handle emotion imbalance

### Machine Learning Models

- **Logistic Regression**: 94.95% accuracy ⭐ (Best performing)
- **Support Vector Machine**: 93.27% accuracy
- **Random Forest**: 91.92% accuracy
- **Naive Bayes**: 83.84% accuracy
- **Ensemble Voting**: 93.94% accuracy

### 1. Data Preprocessing Pipeline

```python
# Text cleaning steps:
1. Lowercase conversion
2. URL removal
3. HTML tag removal
4. Special character removal
5. Stopword removal
6. Lemmatization
```

**Preprocessing Results**:

- Average character reduction: 121.0 characters per text
- Average word reduction: 29.8 words per text

### Model Architecture

#### Currently Implemented: Traditional ML Pipeline

```
Input Text → Preprocessing → TF-IDF Vectorization → Logistic Regression → Emotion Prediction
```

#### Future Enhancement: Bi-LSTM Architecture

```
Input Layer → Embedding(10K vocab, 128 dim) → Bi-LSTM(64 units) → Dropout(0.3) → Dense(64) → Output(softmax)
```

#### Future Enhancement: CNN Model (Ensemble)

```
Input Layer → Embedding(10K vocab, 128 dim) → Conv1D(128, 5) → GlobalMaxPooling → Dense(64) → Output(softmax)
```

### 3. Machine Learning Implementation

#### Feature Engineering

- **TF-IDF Vectorization**: 5,000 features with unigrams and bigrams
- **Preprocessing**: ~44% text size reduction while preserving meaning
- **Class Balancing**: Weighted classes to handle emotion imbalance

#### Model Selection

```
Multiple algorithms tested:
├── Logistic Regression (Winner: 94.95%)
├── Support Vector Machine (93.27%)
├── Random Forest (91.92%)
├── Naive Bayes (83.84%)
└── Ensemble Voting (93.94%)
```

## 📈 Model Performance

### Best Model: Logistic Regression (94.95% Accuracy)

| Emotion      | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| **Anger**    | 1.0000    | 0.5000 | 0.6667   | 2       |
| **Joy**      | 0.9286    | 0.7800 | 0.8478   | 50      |
| **Optimism** | 1.0000    | 1.0000 | 1.0000   | 30      |
| **Sadness**  | 0.9464    | 0.9860 | 0.9658   | 215     |

## 🔍 Key Insights

### Data Characteristics

- **Predominant Emotion**: 72.3% of counseling questions express sadness
- **Topic Focus**: Relationship issues dominate counseling inquiries (25.8%)
- **Class Imbalance**: Anger emotion severely underrepresented (0.5%)

### Technical Findings

- **Preprocessing Impact**: ~44% reduction in text length while preserving meaning
- **Model Efficiency**: Logistic Regression achieves 94.95% accuracy with interpretable features
- **Perfect Classification**: Optimism class shows 100% precision and recall
- **Challenge Area**: Anger detection limited by extremely small sample size (8 cases)

## 🚀 Getting Started

### Quick Start

```bash
git clone https://github.com/your-username/Emotion_Detection_With_Bi_LSTM.git
cd Emotion_Detection_With_Bi_LSTM
pip install -r requirements.txt
jupyter lab
```

📖 **Detailed Setup Guide**: See [SETUP.md](SETUP.md) for complete installation instructions.

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook/Lab
```

### Run the Analysis

1. **Launch Jupyter**: `jupyter lab` or `jupyter notebook`
2. **Open main notebook**: `notebooks/Emotion_Detection_With_Bi_LSTM.ipynb`
3. **Run all cells** to reproduce the complete analysis

### Quick Usage

```python
import pandas as pd

# Load processed data
df = pd.read_csv('data/labeled_emotion_data.csv')

# View emotion distribution
print(df['emotion_label'].value_counts())
```

## 🎯 Applications

### Mental Health Support

- **Automated Triage**: Quickly identify emotional urgency in client messages
- **Therapist Assistance**: Provide emotional context before sessions
- **Progress Tracking**: Monitor emotional journey over therapy sessions
- **Resource Allocation**: Optimize counseling resources based on emotional needs

### Research Applications

- **Emotional Pattern Analysis**: Study common emotional themes in counseling
- **Treatment Effectiveness**: Evaluate therapy outcomes through emotional changes
- **Data-Driven Insights**: Support evidence-based mental health practices

## 🎊 Results Highlights

- **🎯 94.95% Accuracy**: Production-ready model performance
- **🔍 Interpretable Features**: Clear understanding of emotion indicators
- **📊 Rich Visualizations**: Comprehensive data insights and model analysis
- **🧹 Clean Pipeline**: Professional preprocessing and feature engineering
- **📈 Robust Evaluation**: Multiple metrics and cross-validation

## 🚧 Future Enhancements

### Model Improvements

- [ ] **BERT/RoBERTa Fine-tuning**: Advanced transformer models
- [ ] **Multi-label Classification**: Detect multiple emotions simultaneously
- [ ] **Temporal Analysis**: Track emotional changes over time
- [ ] **Deep Learning**: Implement Bi-LSTM and CNN architectures

### Deployment Options

- [ ] **Web Application**: User-friendly interface for therapists
- [ ] **REST API**: Integration with existing mental health systems
- [ ] **Real-time Processing**: Live emotion detection capabilities
- [ ] **Mobile Application**: Smartphone accessibility

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```

### Required Packages

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tensorflow
pip install transformers torch nltk
pip install wordcloud
```

### Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 🚀 Usage

### 1. Data Loading & Exploration

```python
import pandas as pd
df = pd.read_csv('counselchat-data.csv')
```

### 2. Run Preprocessing

```python
# Execute preprocessing cells in the notebook
# This will clean text and generate visualizations
```

### 3. Emotion Labeling

```python
# Automated emotion labeling using transformer models
# Results saved to 'labeled_emotion_data.csv'
```

### 4. Model Training

```python
# Train Bi-LSTM and CNN models
# Ensemble predictions for final results
```

## 📊 Key Visualizations

1. **Text Length Distribution**: Histogram showing character and word count distributions
2. **Topic Analysis**: Bar chart of most common counseling topics
3. **Word Clouds**: Before/after preprocessing text visualization
4. **Emotion Distribution**: Multiple chart types showing emotion label frequencies
5. **Model Performance**: Training/validation accuracy curves

## 🔍 Key Insights

### Data Characteristics

- **Predominant Emotion**: 72.3% of counseling questions express sadness
- **Text Complexity**: Average questions are moderately complex (54 words)
- **Topic Focus**: Relationship issues dominate counseling inquiries (25.8%)
- **Class Imbalance**: Anger emotion severely underrepresented (0.5%)

### Technical Findings

- **Preprocessing Impact**: ~44% reduction in text length while preserving meaning
- **Model Efficiency**: Logistic Regression achieves 94.95% accuracy with interpretable features
- **Perfect Classification**: Optimism class shows 100% precision and recall
- **Challenge Area**: Anger detection limited by extremely small sample size (8 cases)

## 🎯 Applications

### Mental Health Support

- **Automated Triage**: Quickly identify emotional urgency in client messages
- **Therapeutic Insights**: Help therapists understand client emotional states
- **Progress Tracking**: Monitor emotional journey over time

### Research Applications

- **Emotional Pattern Analysis**: Study common emotional themes in counseling
- **Treatment Effectiveness**: Evaluate therapy outcomes through emotional changes
- **Resource Allocation**: Optimize counseling resources based on emotional needs

## 🚧 Future Enhancements

### Model Improvements

- [ ] **BERT/RoBERTa Fine-tuning**: Advanced transformer models
- [ ] **Multi-label Classification**: Detect multiple emotions simultaneously
- [ ] **Temporal Analysis**: Track emotional changes over time
- [ ] **Deep Learning**: Implement Bi-LSTM and CNN architectures

### Deployment Options

- [ ] **Web Application**: User-friendly interface for therapists
- [ ] **REST API**: Integration with existing mental health systems
- [ ] **Real-time Processing**: Live emotion detection capabilities
- [ ] **Mobile Application**: Smartphone accessibility

## 🙏 Acknowledgments

- **Cardiff NLP**: RoBERTa emotion classification model
- **CounselChat**: Original dataset source
- **Hugging Face**: Transformer model infrastructure
- **TensorFlow/Keras**: Deep learning framework

## 📞 Contact

For questions or collaboration opportunities:

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Acknowledgments

- **Cardiff NLP**: RoBERTa emotion classification model
- **CounselChat**: Original dataset source
- **Hugging Face**: Transformer model infrastructure
- **Scikit-learn**: Machine learning framework

---

## ⭐ Star This Repository

If this project helps you or interests you, please consider giving it a star! ⭐

**📧 Questions?** Feel free to open an issue or start a discussion.

---

_Last Updated: July 2025_
