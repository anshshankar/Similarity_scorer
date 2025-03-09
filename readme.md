# Semantic Text Similarity API

A Python-based API for measuring semantic similarity between pairs of text paragraphs.

## Overview

This project provides a solution for quantifying the degree of semantic similarity between two text paragraphs. It uses state-of-the-art NLP techniques to analyze and compare the meaning of texts, returning a similarity score between 0 (completely dissimilar) and 1 (highly similar).

## Features

- Measures semantic similarity between text paragraphs
- Preprocesses text to improve comparison quality
- Provides both single-pair and batch processing capabilities
- Exposes functionality through a RESTful API
- Returns standardized similarity scores between 0-1

## Project Structure

- `semantic_similarity_model.py`: Core model implementation
- `app.py`: Flask API implementation
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container definition

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/semantic-similarity-api.git
cd semantic-similarity-api
pip install -r requirements.txt
```

## Usage

### Using the API Locally

Run the Flask application:

```bash
python app.py
```

Make requests to the API:

```bash
curl -X POST http://localhost:5000/similarity \
  -H "Content-Type: application/json" \
  -d '{"text1": "The quick brown fox jumps over the lazy dog.", "text2": "A fast auburn fox leaps above the sleepy canine."}'
```

Expected response:

```json
{"similarity score": 0.8762}
```

## How It Works

The semantic similarity model uses a pre-trained Sentence Transformer to convert text into high-dimensional vectors (embeddings) that capture semantic meaning. The similarity between texts is calculated using cosine similarity between these vectors:

1. Text preprocessing (lowercasing, removing special characters, stopwords)
2. Generation of text embeddings using a pre-trained transformer model
3. Calculation of cosine similarity between embedding vectors
4. Normalization to a 0-1 scale

## API Reference

### POST /similarity

Calculates the semantic similarity between two text paragraphs.

**Request Format**:
```json
{
  "text1": "First paragraph text...",
  "text2": "Second paragraph text..."
}
```

**Response Format**:
```json
{
  "similarity score": 0.8762
}
```

**Status Codes**:
- 200: Success
- 400: Invalid input (missing text fields)
- 500: Server error

### GET /health

Health check endpoint to verify the API is running.

**Response Format**:
```json
{
  "status": "healthy"
}
```

