import pandas as pd
import numpy as np
import os
import re
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sentence_transformers import SentenceTransformer
from . import config

# Suppress BeautifulSoup warning for URL-like text
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def clean_html(text):
    """Removes HTML tags and entities from text using BeautifulSoup."""
    if not isinstance(text, str) or len(text) == 0:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    clean = soup.get_text(separator=" ")
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def clean_text(text):
    """Basic whitespace normalization."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()

def impute_text(row, priority_cols):
    """
    Smart Imputation: Returns the first non-empty text from priority columns.
    Priority: Description > LongDesc > Title > ProductName > Brand
    """
    for col in priority_cols:
        val = row.get(col, "")
        if isinstance(val, str) and len(val.strip()) > 2:
            return val.strip()
    return ""

def create_text_features(df, text_cols=config.TEXT_COLS):
    """
    Creates 'cluster_text' feature with:
    1. HTML Cleaning
    2. Smart Imputation (no more dropping rows for empty descriptions)
    """
    print(f"Creating features from: {text_cols}")
    print("   > Applying HTML Cleaning...")
    
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").apply(clean_html)
    
    print("   > Applying Smart Imputation (fallback for empty text)...")
    df['cluster_text'] = df.apply(lambda row: impute_text(row, text_cols), axis=1)
    
    df['cluster_text'] = df['cluster_text'].apply(clean_text)
    
    initial_len = len(df)
    df = df[df['cluster_text'].str.len() > 2].copy()
    dropped = initial_len - len(df)
    print(f"Rows after cleaning: {len(df)} (Dropped {dropped}, {dropped/initial_len:.1%})")
    return df

def generate_embeddings(df, model_name=config.EMBEDDING_MODEL):
    """
    Generates or loads embeddings for the 'cluster_text' column.
    """
    cache_file = os.path.join(config.CACHE_DIR, f"embeddings_{model_name}_{len(df)}.npy")
    
    if os.path.exists(cache_file):
        print(f"Loading embeddings from cache: {cache_file}")
        embeddings = np.load(cache_file)
        if len(embeddings) == len(df):
            return embeddings
        else:
            print("Cache size mismatch. Recomputing...")

    print(f"Encoding {len(df)} rows with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['cluster_text'].tolist(), show_progress_bar=True, batch_size=64)

    print(f"Saving embeddings to: {cache_file}")
    np.save(cache_file, embeddings)
    
    return embeddings
