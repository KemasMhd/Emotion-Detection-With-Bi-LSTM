# Data Directory

This directory contains all datasets used in the emotion detection project.

## Files

### Raw Data

- `counselchat-data.csv` - Original counseling chat dataset with 1,482 records
  - **Source**: CounselChat platform
  - **Size**: ~4 MB
  - **Columns**: questionID, questionTitle, questionText, questionUrl, topics, therapistName, therapistUrl, answerText, upvotes

### Processed Data

- `labeled_emotion_data.csv` - Dataset with emotion labels added
  - **Source**: Processed from original dataset using Cardiff NLP RoBERTa model
  - **Emotions**: sadness (72.3%), joy (16.9%), optimism (10.3%), anger (0.5%)
  - **Preprocessing**: Text cleaned, stopwords removed, lemmatized

## Data Usage

```python
import pandas as pd

# Load original data
df_original = pd.read_csv('data/counselchat-data.csv')

# Load processed data with emotion labels
df_processed = pd.read_csv('data/labeled_emotion_data.csv')
```

## Data Privacy

The dataset contains anonymized counseling questions from public sources. No personal information is included.
