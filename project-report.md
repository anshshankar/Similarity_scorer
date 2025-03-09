# Semantic Text Similarity Project Report

## Introduction

This report outlines the approach taken to build and deploy a semantic text similarity model as requested. The project consists of two main parts:
1. Building an algorithm to quantify the semantic similarity between pairs of text paragraphs
2. Deploying this algorithm as an API endpoint on a cloud service provider

## Part A: Semantic Similarity Algorithm

### Core Approach

The semantic similarity model uses a pre-trained Sentence Transformer, specifically the "all-MiniLM-L6-v2" model, which is optimized for computing semantic textual similarity. This approach was chosen for several key reasons:

1. **Effectiveness**: Sentence Transformers are based on state-of-the-art transformer models like BERT and have been fine-tuned specifically for semantic similarity tasks.

2. **Efficiency**: The selected model provides a good balance between performance and computational requirements, making it suitable for deployment in cloud environments.

3. **Robustness**: The model works well with various text lengths and domains, making it appropriate for general-purpose semantic similarity measurement.

The algorithm follows these steps:

1. **Text Preprocessing**:
   - Convert to lowercase
   - Remove special characters
   - Remove stopwords
   - Normalize whitespace

2. **Embedding Generation**:
   - The preprocessed texts are passed through the Sentence Transformer model
   - This generates dense vector representations (embeddings) that capture the semantic meaning of each text

3. **Similarity Computation**:
   - Cosine similarity is calculated between the embedding vectors
   - The result is a value between -1 and 1, which is then normalized to a 0-1 scale
   - 1 represents high similarity, 0 represents low similarity

This approach does not require labeled data, making it suitable for the unsupervised nature of the problem. The pre-trained model has already learned semantic relationships from vast amounts of text data, allowing it to generalize well to new text pairs.

### Model Performance Considerations

While no labeled test data was provided to evaluate the model quantitatively, several qualitative tests were performed with various text pairs. The model demonstrates good discrimination between:
- Texts with similar topics but different meanings
- Texts with similar wording but different contexts
- Texts with completely different topics

The semantic similarity scores align well with human intuition in most cases, making the model suitable for the intended application.

## Part B: API Deployment

### Core Approach

The model was wrapped in a Flask API and deployed as a containerized application. This approach was chosen for:

1. **Portability**: Containerization allows the application to run consistently across different environments
2. **Scalability**: Cloud platforms can easily scale containerized applications based on demand
3. **Simplicity**: Flask provides a lightweight framework that's easy to deploy and maintain

The API deployment follows these steps:

1. **API Development**:
   - Created a Flask application with a `/similarity` endpoint
   - Implemented proper error handling and input validation
   - Added a health check endpoint for monitoring

2. **Containerization**:
   - Defined a Dockerfile to package the application and its dependencies
   - Optimized the container size by using a slim Python base image
   - Pre-downloaded NLTK resources during image building

3. **Cloud Deployment**:
   - Deployed the container to a cloud service provider
   - Configured appropriate resource allocation
   - Set up logging and monitoring

The API follows the exact request-response format specified in the requirements:

Request:
```json
{
  "text1": "nuclear body seeks new tech.......",
  "text2": "terror suspects face arrest........"
}
```

Response:
```json
{
  "similarity score": 0.2
}
```

### Deployment Considerations

The API was designed with several important considerations:

1. **Performance**: The model is loaded once during startup to avoid reloading it for each request
2. **Error Handling**: Proper validation and error messages are provided for invalid inputs
3. **Scalability**: The application can handle multiple concurrent requests
4. **Monitoring**: A health check endpoint allows for uptime monitoring

