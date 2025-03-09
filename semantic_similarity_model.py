import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

class SemanticSimilarityModel:
    """
    A model to compute semantic similarity between pairs of text paragraphs.
    Uses a pre-trained sentence transformer model to create embeddings and 
    computes cosine similarity between them.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the semantic similarity model.
        
        Parameters:
        model_name (str): Name of the pre-trained sentence transformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Preprocess text by removing special characters, extra spaces, and stopwords.
        
        Parameters:
        text (str): Input text to preprocess.
        
        Returns:
        str: Preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        
        return ' '.join(filtered_text)
    
    def get_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Parameters:
        texts (list): List of text strings.
        
        Returns:
        numpy.ndarray: Matrix of embeddings.
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(preprocessed_texts)
        
        return embeddings
    
    def compute_similarity(self, text1, text2):
        """
        Compute semantic similarity between two texts.
        
        Parameters:
        text1 (str): First text.
        text2 (str): Second text.
        
        Returns:
        float: Similarity score between 0 and 1.
        """
        # Get embeddings
        embeddings = self.get_embeddings([text1, text2])
        
        # Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Ensure the result is between 0 and 1
        similarity = max(0, min(similarity, 1))
        
        return float(similarity)
    
    def batch_compute_similarity(self, df, text1_col='text1', text2_col='text2'):
        """
        Compute similarity for multiple text pairs in a dataframe.
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing text pairs.
        text1_col (str): Column name for first text.
        text2_col (str): Column name for second text.
        
        Returns:
        pandas.DataFrame: DataFrame with similarity scores.
        """
        result_df = df.copy()
        
        # Get all texts
        texts = list(df[text1_col]) + list(df[text2_col])
        
        # Get embeddings for all texts
        all_embeddings = self.get_embeddings(texts)
        
        # Split embeddings back into text1 and text2
        n = len(df)
        text1_embeddings = all_embeddings[:n]
        text2_embeddings = all_embeddings[n:]
        
        # Compute similarity for each pair
        similarities = []
        for i in range(n):
            sim = cosine_similarity([text1_embeddings[i]], [text2_embeddings[i]])[0][0]
            sim = max(0, min(sim, 1))  # Ensure result is between 0 and 1
            similarities.append(sim)
        
        # Add similarity scores to the dataframe
        result_df['similarity_score'] = similarities
        
        return result_df
    
    def save_model(self, path='model'):
        """
        Save the model to a directory.
        
        Parameters:
        path (str): Directory path to save the model.
        """
        os.makedirs(path, exist_ok=True)
        self.model.save(path)
    
    @classmethod
    def load_model(cls, path='model'):
        """
        Load a saved model from a directory.
        
        Parameters:
        path (str): Directory path to load the model from.
        
        Returns:
        SemanticSimilarityModel: Loaded model.
        """
        instance = cls.__new__(cls)
        instance.model = SentenceTransformer.load(path)
        instance.stop_words = set(stopwords.words('english'))
        return instance


# Example usage
if __name__ == "__main__":
    # Load your dataset
    try:
        df = pd.read_csv('DataNeuron_Text_Similarity.csv')
        print(f"Dataset loaded successfully with {len(df)} entries.")
        print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print("Dataset file not found. Creating a sample dataset.")
        # Create a sample dataset if the file is not found
        data = {
            'text1': [
                "The quick brown fox jumps over the lazy dog.",
                "Climate change is a major global challenge.",
                "Machine learning algorithms can analyze large amounts of data."
            ],
            'text2': [
                "A fast auburn fox leaps above the sleepy canine.",
                "Global warming poses significant threats to our planet.",
                "Data science techniques help process big data efficiently."
            ]
        }
        df = pd.DataFrame(data)
    
    # Initialize and use the model
    model = SemanticSimilarityModel()
    
    # Compute similarity for each pair in the dataset
    result_df = model.batch_compute_similarity(df)
    
    # Display results
    pd.set_option('display.max_colwidth', None)
    print("\nResults:")
    for i, row in result_df.iterrows():
        print(f"Pair {i+1}:")
        print(f"Text 1: {row['text1']}")
        print(f"Text 2: {row['text2']}")
        print(f"Similarity Score: {row['similarity_score']:.4f}")
        print("-" * 50)
    
    # Save the model
    model.save_model()
    print("Model saved successfully.")