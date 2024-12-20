# Semantic Similarity of Words, Phrases, and Sentences

## Overview
This repository contains the implementation and evaluation of various approaches to compute semantic similarity in Natural Language Processing (NLP). The tasks involve computing the similarity between words, phrases, and sentences using techniques ranging from traditional word embeddings to advanced transformer models.

## Features
- **Word Embeddings:** Implementation of Continuous Bag of Words (CBOW) and Skip-Gram models from scratch using the Brown Corpus.
- **Pre-trained Models:** Utilization of SpaCy's pre-trained embeddings for word similarity.
- **Sentence and Phrase Similarity:** 
  - Average of word embeddings for sentence representation.
  - Cosine similarity for quantifying semantic similarity.
- **Fine-Tuned Transformers:** 
  - BERT fine-tuned on the PAWS dataset for sentence similarity.

## Datasets
1. **Brown Corpus** - Used for training CBOW and Skip-Gram models.
2. **SimLex-999** - Evaluated word similarity models.
3. **PAWS Dataset** - Used for training and evaluating sentence similarity tasks.

## Requirements
- Python 3.8+
- Libraries:
  - `torch`
  - `transformers`
  - `spacy`
  - `sklearn`
  - `datasets`
  - `tqdm`
- SpaCy model: `en_core_web_md`

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/[YourUsername]/semantic-similarity-nlp.git
   cd semantic-similarity-nlp
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download SpaCy model:
   ```bash
   python -m spacy download en_core_web_md
   ```
4. Run the Individual Python Notebooks for each task

## Implementation Details
### Word Embeddings from Scratch
1. **CBOW Model**:
   - Predicts a word based on surrounding context.
   - Trained using cross-entropy loss and evaluated on SimLex-999.
2. **Skip-Gram Model**:
   - Predicts context words from a target word.
   - Focuses on rare word representation.

### Sentence and Phrase Similarity
- Sentence embeddings are computed as the average of word embeddings.
- Cosine similarity is used for quantifying semantic similarity between pairs of sentences or phrases.

### Fine-Tuned BERT
- BERT fine-tuned using the PAWS dataset for binary classification of sentence similarity.
- Achieves state-of-the-art results in accuracy, precision, recall, and F1 score.

## Results
| Model            | Accuracy | Precision | Recall | F1 Score | Correlation |
|------------------|----------|-----------|--------|----------|-------------|
| **CBOW**        | 0.5085   | 0.4498    | 0.5079 | 0.4771   | 0.0239      |
| **Skip-Gram**   | 0.5085   | 0.4384    | 0.4036 | 0.4203   | 0.0200      |
| **SpaCy (Word)**| 0.5816   | 0.5401    | 0.3515 | 0.4258   | 0.2275      |
| **Phrase Avg.** | 0.5040   | 0.5036    | 0.5650 | 0.5325   | 0.0386      |
| **Sentence Avg.**| 0.4434  | 0.4422    | 0.9924 | 0.6118   | -0.0155     |
| **Fine-Tuned BERT**| 0.9019 | 0.8585    | 0.9316 | 0.8935   | 0.8374      |


## References
- [Word2Vec From Scratch](https://medium.com/@enozeren/word2vec-from-scratch-with-python-1bba88d9f221)
- [Mastering NLP with GloVe Embeddings](https://muneebsa.medium.com/mastering-nlp-with-glove-embeddings-word-similarity-sentiment-analysis-and-more-27f731988c48)
- [PAWS Dataset](https://huggingface.co/datasets/google-research-datasets/paws)
- [CBOW vs Skip-Gram](https://www.geeksforgeeks.org/word-embeddings-in-nlp-comparison-between-cbow-and-skip-gram-models/)
