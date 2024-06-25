import os
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pickle
from tqdm import tqdm
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_pdfs_from_directory(directory, text_dir):
    """
    Loads all PDF documents from the specified directory, extracts text, and saves text to files.
    
    Args:
        directory (str): Path to the directory containing PDF files.
        text_dir (str): Path to the directory where text files will be saved.
    
    Returns:
        texts (list): List of extracted text from each PDF.
    """
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    texts = []
    for filename in tqdm(os.listdir(directory), desc=f'Loading PDFs from {directory}'):
        if filename.endswith('.pdf'):
            text_filepath = os.path.join(text_dir, filename.replace('.pdf', '.txt'))
            if os.path.exists(text_filepath):
                with open(text_filepath, 'r') as f:
                    text = f.read()
            else:
                filepath = os.path.join(directory, filename)
                text = extract_text_from_pdf(filepath)
                if text:
                    with open(text_filepath, 'w') as f:
                        f.write(text)
            if text:
                texts.append(text)
    return texts

def extract_text_from_pdf(filepath):
    """
    Extracts text from a PDF file.
    
    Args:
        filepath (str): Path to the PDF file.
    
    Returns:
        text (str): Extracted text or None if extraction fails.
    """
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        return None

def preprocess(text):
    """
    Preprocesses the text by tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text (str): The text to preprocess.
    
    Returns:
        tokens (list): List of processed tokens.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

def train_lda_model(processed_corpus, num_topics=10, passes=15):
    """
    Trains the LDA model on the processed corpus.
    
    Args:
        processed_corpus (list): List of processed documents.
        num_topics (int): Number of topics to extract.
        passes (int): Number of passes through the corpus during training.
    
    Returns:
        lda_model (LdaModel): Trained LDA model.
        dictionary (Dictionary): Dictionary created from the corpus.
        doc_term_matrix (list): Document-term matrix for the corpus.
    """
    dictionary = corpora.Dictionary(processed_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_corpus]
    lda_model = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes)
    return lda_model, dictionary, doc_term_matrix

def evaluate_lda_model(lda_model, processed_corpus, dictionary, lambda_value=0.3, sample_size=100):
    """
    Evaluates the LDA model using coherence score.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        processed_corpus (list): List of processed documents.
        dictionary (Dictionary): Dictionary created from the corpus.
        lambda_value (float): Lambda value for coherence score calculation.
        sample_size (int): Number of documents to sample for coherence calculation.
    
    Returns:
        coherence_lda (float): Coherence score of the LDA model.
    """
    sample_texts = random.sample(processed_corpus, min(sample_size, len(processed_corpus)))
    coherence_model_lda = CoherenceModel(model=lda_model, texts=sample_texts, dictionary=dictionary, coherence='c_v', topn=int(lambda_value * len(dictionary)))
    
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda

def save_model_and_data(lda_model, dictionary, tfidf_vectorizer, tfidf_matrix, file_prefix):
    """
    Saves the LDA model, dictionary, TF-IDF vectorizer, and TF-IDF matrix to disk.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        dictionary (Dictionary): Dictionary created from the corpus.
        tfidf_vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): TF-IDF matrix of the corpus.
        file_prefix (str): Prefix for the saved file names.
    """
    with open(f'{file_prefix}_lda_model.pkl', 'wb') as f:
        pickle.dump(lda_model, f)
    with open(f'{file_prefix}_dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
    with open(f'{file_prefix}_tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(f'{file_prefix}_tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)

def main_training():
    # Directories containing PDF files and where text files will be saved
    directories = ['./civicplus_docs', './civicplus_docs-2', './civicplus_docs-3']
    text_dir = './text_files'
    
    # Load data from all directories
    print("Loading PDF files...")
    corpus = []
    for directory in directories:
        corpus.extend(load_pdfs_from_directory(directory, text_dir))
    
    # Preprocess data
    print("Preprocessing documents...")
    processed_corpus = [preprocess(doc) for doc in tqdm(corpus, desc='Preprocessing')]
    
    # Hyperparameters
    num_topics = 10
    passes = 15
    lambda_value = 0.3
    
    # Train LDA model
    print("Training LDA model...")
    lda_model, dictionary, doc_term_matrix = train_lda_model(processed_corpus, num_topics, passes)
    
    # Evaluate LDA model
    print("Evaluating LDA model...")
    coherence_lda = evaluate_lda_model(lda_model, processed_corpus, dictionary, lambda_value, sample_size=100)
    print(f'Coherence Score: {coherence_lda}')
    
    # Compute TF-IDF
    print("Computing TF-IDF matrix...")
    flattened_corpus = [' '.join(doc) for doc in tqdm(processed_corpus, desc='Flattening corpus')]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(flattened_corpus)
    
    # Save models and data
    print("Saving models and data...")
    save_model_and_data(lda_model, dictionary, tfidf_vectorizer, tfidf_matrix, 'agenda_watch')

if __name__ == "__main__":
    main_training()
