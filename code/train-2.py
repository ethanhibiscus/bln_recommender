import os
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora, models
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

warnings.filterwarnings('ignore')

# Download NLTK resources once
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
        filenames (list): List of filenames corresponding to the extracted texts.
    """
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    texts = []
    filenames = []
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
                filenames.append(filename)
    return texts, filenames

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

def visualize_token_representations(processed_corpus, output_dir):
    """
    Visualizes token representations using word clouds and bar plots.
    
    Args:
        processed_corpus (list): List of processed documents.
        output_dir (str): Directory where the visualizations will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_tokens = [token for doc in processed_corpus for token in doc]
    word_freq = nltk.FreqDist(all_tokens)
    
    # Word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Tokens")
    plt.savefig(os.path.join(output_dir, "word_cloud_tokens.png"))
    plt.close()
    
    # Bar plot
    most_common_words = word_freq.most_common(20)
    words, frequencies = zip(*most_common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=frequencies, y=words)
    plt.title("Top 20 Most Common Words")
    plt.savefig(os.path.join(output_dir, "bar_plot_tokens.png"))
    plt.close()

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
    lda_model = models.LdaMulticore(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes)
    return lda_model, dictionary, doc_term_matrix

def visualize_lda_model(lda_model, dictionary, doc_term_matrix, output_dir):
    """
    Visualizes the LDA model using pyLDAvis and saves it as an HTML file.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        dictionary (Dictionary): Dictionary created from the corpus.
        doc_term_matrix (list): Document-term matrix for the corpus.
        output_dir (str): Directory where the visualization will be saved.
    """
    vis_data = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)
    pyLDAvis.save_html(vis_data, os.path.join(output_dir, 'lda_visualization.html'))

def generate_document_topic_matrix(lda_model, doc_term_matrix, filenames):
    """
    Generates the document-topic matrix.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        doc_term_matrix (list): Document-term matrix for the corpus.
        filenames (list): List of filenames corresponding to the documents.
    
    Returns:
        doc2topic (DataFrame): Document-topic matrix.
    """
    doc2topic = lda_model.get_document_topics(doc_term_matrix, minimum_probability=0)
    doc2topic = pd.DataFrame(list(doc2topic))
    num_topics = lda_model.num_topics
    doc2topic.columns = ['Topic'+str(i+1) for i in range(num_topics)]
    for i in range(len(doc2topic.columns)):
        doc2topic.iloc[:,i] = doc2topic.iloc[:,i].apply(lambda x: x[1])
    doc2topic['Automated_topic_id'] = doc2topic.apply(lambda x: np.argmax(x), axis=1)
    doc2topic['Filename'] = filenames
    return doc2topic

def save_model_and_data(lda_model, dictionary, tfidf_vectorizer, tfidf_matrix, doc2topic, file_prefix, output_dir):
    """
    Saves the LDA model, dictionary, TF-IDF vectorizer, TF-IDF matrix, and document-topic matrix to disk.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        dictionary (Dictionary): Dictionary created from the corpus.
        tfidf_vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): TF-IDF matrix of the corpus.
        doc2topic (DataFrame): Document-topic matrix.
        file_prefix (str): Prefix for the saved file names.
        output_dir (str): Directory where the models and data will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f'{file_prefix}_lda_model.pkl'), 'wb') as f:
        pickle.dump(lda_model, f)
    with open(os.path.join(output_dir, f'{file_prefix}_dictionary.pkl'), 'wb') as f:
        pickle.dump(dictionary, f)
    with open(os.path.join(output_dir, f'{file_prefix}_tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(output_dir, f'{file_prefix}_tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    doc2topic.to_csv(os.path.join(output_dir, f'{file_prefix}_doc2topic.csv'), index=False)

def main_training():
    # Directories containing PDF files and where text files will be saved
    directories = ['./civicplus_docs', './civicplus_docs-2', './civicplus_docs-3']
    text_dir = './text_files'
    output_dir = 'train_output'
    
    # Load data from all directories
    print("Loading PDF files...")
    corpus = []
    filenames = []
    for directory in directories:
        texts, files = load_pdfs_from_directory(directory, text_dir)
        corpus.extend(texts)
        filenames.extend(files)
    
    # Preprocess data
    print("Preprocessing documents...")
    processed_corpus = [preprocess(doc) for doc in tqdm(corpus, desc='Preprocessing')]
    
    # Visualize token representations
    print("Visualizing token representations...")
    visualize_token_representations(processed_corpus, output_dir)
    
    # Hyperparameters
    num_topics = 10
    passes = 15
    
    # Train LDA model
    print("Training LDA model...")
    lda_model, dictionary, doc_term_matrix = train_lda_model(processed_corpus, num_topics, passes)
    
    # Visualize LDA model
    print("Visualizing LDA model...")
    visualize_lda_model(lda_model, dictionary, doc_term_matrix, output_dir)
    
    # Generate document-topic matrix
    print("Generating document-topic matrix...")
    doc2topic = generate_document_topic_matrix(lda_model, doc_term_matrix, filenames)
    
    # Compute TF-IDF
    print("Computing TF-IDF matrix...")
    flattened_corpus = [' '.join(doc) for doc in tqdm(processed_corpus, desc='Flattening corpus')]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(flattened_corpus)
    
    # Save models and data
    print("Saving models and data...")
    save_model_and_data(lda_model, dictionary, tfidf_vectorizer, tfidf_matrix, doc2topic, 'agenda_watch', output_dir)

if __name__ == "__main__":
    main_training()
