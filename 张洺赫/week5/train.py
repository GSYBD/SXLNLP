import numpy as np
import pandas as pd

from gensim.models import Word2Vec

import nltk
from nltk import word_tokenize
from nltk.cluster import KMeansClusterer
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")

def clean_text(text, tokenizer, stopwords):
    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens

def read_data(path):
    df = pd.read_csv(path)

    custom_stopwords = set(stopwords.words("english") + ["news", "new", "top"])
    text_columns = ["title", "description", "content"]
    
    for col in text_columns:
        df[col] = df[col].astype(str)

    df["text"] = df[text_columns].apply(lambda x: " | ".join(x), axis=1)
    df["tokens"] = df["text"].map(lambda x: clean_text(x, word_tokenize, custom_stopwords))

    tokenized_docs = df["tokens"].values
    return tokenized_docs

def get_vector_model(tokenized_docs):
    model = Word2Vec(sentences=tokenized_docs, vector_size=100, workers=8)
    return model

def vectorize(list_of_docs, model):
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

def Kclusters(X, num_means=3):
    kclusterer = KMeansClusterer(num_means=num_means, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    return assigned_clusters

def main():
   tokenized_docs = read_data("./news_data.csv")
   model = get_vector_model(tokenized_docs)
   vectors = vectorize(tokenized_docs, model=model)
   clusters = Kclusters(vectors, num_means=50)
   print(clusters[:10])

    
if __name__ == "__main__":
    main()