import os
import shutil
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import random

def load_model_and_data(file_prefix):
    """
    Loads the LDA model, dictionary, TF-IDF vectorizer, and TF-IDF matrix from disk.
    
    Args:
        file_prefix (str): Prefix for the saved file names.
    
    Returns:
        lda_model (LdaModel): Loaded LDA model.
        dictionary (Dictionary): Loaded dictionary.
        tfidf_vectorizer (TfidfVectorizer): Loaded TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): Loaded TF-IDF matrix.
        doc2topic (DataFrame): Document-topic matrix.
    """
    output_dir = 'train_output'
    with open(os.path.join(output_dir, f'{file_prefix}_lda_model.pkl'), 'rb') as f:
        lda_model = pickle.load(f)
    with open(os.path.join(output_dir, f'{file_prefix}_dictionary.pkl'), 'rb') as f:
        dictionary = pickle.load(f)
    with open(os.path.join(output_dir, f'{file_prefix}_tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(os.path.join(output_dir, f'{file_prefix}_tfidf_matrix.pkl'), 'rb') as f:
        tfidf_matrix = pickle.load(f)
    doc2topic = pd.read_csv(os.path.join(output_dir, f'{file_prefix}_doc2topic.csv'))
    return lda_model, dictionary, tfidf_vectorizer, tfidf_matrix, doc2topic

def compute_cosine_similarity_within_topic(tfidf_matrix, doc_index, topic_index, doc2topic):
    """
    Computes cosine similarity between the selected document and all other documents within the same topic.
    
    Args:
        tfidf_matrix (sparse matrix): TF-IDF matrix of the corpus.
        doc_index (int): Index of the selected document.
        topic_index (int): Topic index of the selected document.
        doc2topic (DataFrame): Document-topic matrix.
    
    Returns:
        cosine_similarities (array): Array of cosine similarity scores within the same topic.
        same_topic_docs (Index): Indices of documents within the same topic.
    """
    same_topic_docs = doc2topic[doc2topic['Automated_topic_id'] == topic_index].index
    cosine_similarities = cosine_similarity(tfidf_matrix[doc_index], tfidf_matrix[same_topic_docs]).flatten()
    return cosine_similarities, same_topic_docs

def get_top_n_recommendations(cosine_similarities, same_topic_docs, N=3):
    """
    Gets the top N recommended documents based on cosine similarity scores.
    
    Args:
        cosine_similarities (array): Array of cosine similarity scores.
        same_topic_docs (Index): Indices of documents within the same topic.
        N (int): Number of recommendations to retrieve.
    
    Returns:
        recommended_doc_indices (list): List of indices of the recommended documents.
    """
    ranked_doc_indices = cosine_similarities.argsort()[::-1][:N]
    recommended_doc_indices = same_topic_docs[ranked_doc_indices].tolist()
    return recommended_doc_indices

def save_recommended_docs(doc_indices, selected_doc_index, doc2topic, source_dirs, destination_base_dir):
    """
    Saves the top N recommended documents and the selected document to a new folder.
    
    Args:
        doc_indices (list): List of indices of the recommended documents.
        selected_doc_index (int): Index of the selected document.
        doc2topic (DataFrame): Document-topic matrix.
        source_dirs (list): List of directories containing the original PDF files.
        destination_base_dir (str): Base directory where the new folder will be created.
    """
    folder_index = 0
    destination_dir = f"{destination_base_dir}/recommendations_{folder_index}"
    
    while os.path.exists(destination_dir):
        folder_index += 1
        destination_dir = f"{destination_base_dir}/recommendations_{folder_index}"
    
    os.makedirs(destination_dir)
    
    all_indices = [selected_doc_index] + doc_indices
    print(f"Indices to save: {all_indices}")
    
    for idx in all_indices:
        filename = doc2topic.loc[idx, 'Filename']
        print(f"Attempting to find file: {filename}")
        file_found = False
        for source_dir in source_dirs:
            source_filepath = os.path.join(source_dir, filename)
            if os.path.exists(source_filepath):
                destination_filepath = os.path.join(destination_dir, filename)
                shutil.copy(source_filepath, destination_filepath)
                print(f"Copied {filename} from {source_dir} to {destination_dir}")
                file_found = True
                break
        if not file_found:
            print(f"File {filename} not found in any source directories")
    
    print(f"Documents saved to {destination_dir}")

def main_inference(selected_doc_name=None, N=3):
    # Load models and data
    print("Loading models and data...")
    lda_model, dictionary, tfidf_vectorizer, tfidf_matrix, doc2topic = load_model_and_data('agenda_watch')
    
    if selected_doc_name == "random":
        selected_doc_index = doc2topic.sample().index[0]
        selected_doc_name = doc2topic.loc[selected_doc_index, 'Filename']
        print(f"Randomly selected document: {selected_doc_name} (Index: {selected_doc_index})")
    else:
        selected_doc_index = doc2topic[doc2topic['Filename'] == selected_doc_name].index[0]
        print(f"Selected document: {selected_doc_name} (Index: {selected_doc_index})")
    
    # Get the topic of the selected document
    selected_doc_topic = doc2topic.loc[selected_doc_index, 'Automated_topic_id']
    print(f"Selected document topic: {selected_doc_topic}")
    
    # Compute similarity scores within the same topic
    similarity_scores, same_topic_docs = compute_cosine_similarity_within_topic(tfidf_matrix, selected_doc_index, selected_doc_topic, doc2topic)
    
    # Get top N recommendations
    recommended_doc_indices = get_top_n_recommendations(similarity_scores, same_topic_docs, N)
    print(f"Top {N} recommended document indices: {recommended_doc_indices}")
    
    # Save recommended documents
    save_recommended_docs(recommended_doc_indices, selected_doc_index, doc2topic, ['./civicplus_docs', './civicplus_docs-2', './civicplus_docs-3'], './recommendations')
    
    return recommended_doc_indices, similarity_scores, doc2topic

if __name__ == "__main__":
    # User input for the selected document name
    selected_doc_name = input("Enter the name of the document (or 'random' for a random document): ")
    recommended_doc_indices, similarity_scores, doc2topic = main_inference(selected_doc_name)
    
    # Print recommended documents
    for idx in recommended_doc_indices:
        print(f'Document {doc2topic.loc[idx, "Filename"]} with similarity score {similarity_scores[idx]}')
