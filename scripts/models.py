# NLP models, loaded once at startup and reused throughout the pipeline.
#
# - spaCy `it_core_news_lg` for Italian NER and lemmatization
# - `paraphrase-multilingual-mpnet-base-v2` for semantic embeddings
# - NLTK stopwords and tokenizer for Italian
#
# Imports trigger downloads on first run. Subsequent imports are cheap
# because spaCy and SentenceTransformer cache the loaded models in process.

import nltk
import spacy
import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer


# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"


# Italian spaCy model (large), used for NER and lemmatization.
# Run once to download:  python -m spacy download it_core_news_lg
nlp_it = spacy.load("it_core_news_lg")


# Multilingual sentence transformer for semantic embeddings
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
).to(device)


# NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

stop_words_italian = set(stopwords.words("italian"))
