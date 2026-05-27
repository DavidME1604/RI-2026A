import kagglehub
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer_english = SnowballStemmer('english')


def download_dataset() -> str:
    path = kagglehub.dataset_download("thedevastator/uncovering-financial-insights-with-the-reuters-2")
    return path

def process_raw(raw_text: str) -> list:
    clean_text = re.sub(r'[^a-z\s]', '', raw_text)
    tokens = word_tokenize(clean_text.lower())
    stemmed_words = [stemmer_english.stem(token) for token in tokens if token not in stop_words] 
    return stemmed_words

def process_corpus(path: str)-> pd.DataFrame:
    corpus = {'doc_id': [], 'title': [], 'raw': [], 'tokenized': [],'processed': []}
    df_input = pd.read_csv(path)
    for id_doc, row in df_input.iterrows():
        text = str(row['title'])+' '+str(row['text'])
        stemmed_words = process_raw(text)
        corpus['doc_id'].append(id_doc)
        corpus['title'].append(row['title'])
        corpus['raw'].append(text)
        corpus['tokenized'].append(stemmed_words)
        corpus['processed'].append(' '.join(stemmed_words))
    return pd.DataFrame(corpus)

def inverse_index(df_corpus: pd.DataFrame) -> dict:
    index = {'docs': {}, 'freq': {}} 
    
    for _, row in df_corpus.iterrows():
        doc_id = row['doc_id']
        conteo_terminos = Counter(row['tokenized'])
        for term, frecuencia_en_doc in conteo_terminos.items():
            if term not in index['docs']:
                index['docs'][term] = {} 
            index['docs'][term][doc_id] = frecuencia_en_doc
            if term not in index['freq']:
                index['freq'][term] = 0                
            index['freq'][term] += frecuencia_en_doc

    return index

def build_binary_matrix(df_corpus: pd.DataFrame):
    binary_vectorizer = CountVectorizer(binary=True)
    binary_matrix = binary_vectorizer.fit_transform(df_corpus['processed'])
    return binary_matrix, binary_vectorizer

def jaccard_similarity(query_processed: str, corpus_matrix, vectorizer):
    query_vector = vectorizer.transform([query_processed])
    interseccion = corpus_matrix.dot(query_vector.T)
    union = np.array(corpus_matrix.sum(axis=1) + query_vector.sum() - interseccion).flatten()
    interseccion = np.array(interseccion.toarray()).flatten()
    scores = np.divide(interseccion, union, out=np.zeros_like(interseccion, dtype=float), where=union != 0)
    ranking = scores.argsort()[::-1]
    return ranking, scores

def build_tfidf_matrix(df_corpus: pd.DataFrame):
    vectorizer_TFIDF = TfidfVectorizer(lowercase=False)
    tfidf_matrix = vectorizer_TFIDF.fit_transform(df_corpus['processed'])
    return tfidf_matrix, vectorizer_TFIDF

def tf_idf_similarity(query_processed, corpus_matrix, vectorizer_TFIDF):
    query_vector = vectorizer_TFIDF.transform([query_processed])
    scores = cosine_similarity(query_vector, corpus_matrix).flatten()
    ranking = scores.argsort()[::-1]
    return ranking, scores

def process_query(query_text: str) -> dict:
    tokens = process_raw(query_text)
    processed = ' '.join(tokens)
    return {
        'raw': query_text,
        'tokenized': tokens,
        'processed': processed,
    }

def show_results(query: dict, ranking, scores, df_corpus: pd.DataFrame, model_name: str, top_n: int = 5):
    print(f"{'='*80}")
    print(f"  Modelo: {model_name}")
    print(f"  Query: {query['raw']}")
    print(f"  Tokens procesados: {query['tokenized']}")
    print(f"{'='*80}")
    print(f"  {'Rank':<6} {'Doc ID':<10} {'Score':<12} {'Título'}")
    print(f"  {'-'*70}")
    for i, idx in enumerate(ranking[:top_n]):
        print(f"  {i+1:<6} {df_corpus.loc[idx, 'doc_id']:<10} {scores[idx]:<12.4f} {df_corpus.loc[idx, 'title']}")
    print()


def main():
    print("Cargando dataset...")
    path = download_dataset()
    df_corpus = process_corpus(path + "/ModApte_train.csv")
    
    print("Construyendo índices...")
    binary_matrix, binary_vectorizer = build_binary_matrix(df_corpus)
    tfidf_matrix, vectorizer_tfidf = build_tfidf_matrix(df_corpus)
    inv_index = inverse_index(df_corpus)
    
    modelos = {
        '1': 'Jaccard',
        '2': 'TF-IDF Coseno',
        '3': 'Todos'
    }
    
    print("\nSistema de Recuperación de Información")
    print("Escriba 'salir' para terminar.\n")
    
    while True:
        query_text = input("Ingrese su consulta: ").strip()
        if query_text.lower() == 'salir':
            print("Hasta luego!")
            break
        if not query_text:
            print("La consulta no puede estar vacía.\n")
            continue
        
        print("\nModelos disponibles:")
        for key, nombre in modelos.items():
            print(f"  {key}. {nombre}")
        
        opcion = input("Seleccione modelo: ").strip()
        if opcion not in modelos:
            print("Opción no válida.\n")
            continue
        
        try:
            processed = process_query(query_text)
            
            if opcion in ('1', '3'):
                ranking, scores = jaccard_similarity(processed['processed'], binary_matrix, binary_vectorizer)
                show_results(processed, ranking, scores, df_corpus, "Jaccard")
            
            if opcion in ('2', '3'):
                ranking, scores = tf_idf_similarity(processed['processed'], tfidf_matrix, vectorizer_tfidf)
                show_results(processed, ranking, scores, df_corpus, "TF-IDF Coseno")
        
        except Exception as e:
            print(f"Error procesando consulta: {e}\n")

if __name__ == "__main__":
    main()

