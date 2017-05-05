"""
Created on Sun Apr 09 21:39:20 2017
@title: classifying fake news
@author: Brynn Arborico and Alice Szeliga
"""
# python libraries
import collections

# numpy libraries
import numpy as np

# natural language toolkit
# http://www.nltk.org/
import nltk

# k-modes and k-prototypes
#from kmodesMaster.kmodes.kprototypes import *

# libraries specific to project
from util import *
#from cluster import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def extract_feature_vectors(X, text_fields, numerical_fields):
    """
    Extracts a feature matrix from X, treating data as text (processed as a bag
    of words), categorical (left as a single string), or numerical (converted
    to float).
    
    Parameters
    --------------------
        X                -- ndarray of dimensions (n,d), data samples
        text_fields      -- list of indices corresponding to textual fields
                            (columns in X)
        numerical_fields -- list of indices corresponding to numerical fields
                            (columns in X)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d), each feature is either
                          the original field value (for categorical), the float
                          interpretation (for numerical), or a TF-IDF bag-of-
                          words representation (for text)
        vectorizer     -- the CountVectorizer used
        transformer    -- the TfidfTransformer used
    """
    vectorizer = CountVectorizer(stop_words='english', lowercase=False, min_df=1, max_features=None)
    transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
    
    n,d = X.shape
    feature_matrix = np.empty((n, 0), dtype=object)
    for field in range(d):
        if field in text_fields:
            field_text = X[:, field]
            counts = vectorizer.fit_transform(field_text)
            tfidf = transformer.fit_transform(counts)
            print "tfidf:", tfidf.toarray()
            feature_matrix = np.concatenate((feature_matrix, tfidf.toarray()), axis=1)
            
        else:
            field_values = X[:, field]
            if field in numerical_fields:
                field_values = field_values.astype(np.float)
            feature_matrix = np.concatenate((feature_matrix, field_values.reshape(n, 1)), axis=1)
            
    return feature_matrix, vectorizer, transformer


def calculate_purity(clusters, weighted=False):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        clusters    -- output of kprototypes fit_predict function, the set of clusters
        weighted    -- bool, whether to weight the purities of each cluster when averaging
    
    Returns
    --------------------
        overall_purity -- a float between 0 and 1 representing the averaged purity of the clusters
    """
    overall_purity = 0
    # TODO: not sure this is how clusters is structured
    for cluster in clusters:
        labels = []
        for p in cluster:
            labels.append(p.label)
        
        cluster_label, count = stats.mode(labels)
    


def main():
    
    np.set_printoptions(threshold=np.nan)
    
    # Fetch stop words and tokenizer
    # download Stopwords Corpus and Punkt Tokenizer Models
    # from http://www.nltk.org/nltk_data/
    # unzip folder into another called "corpora" or "tokenizers", respectively,
    # at one of the following locations:
    #    - 'C:\\Users\\brynn/nltk_data'
    #    - 'C:\\nltk_data'
    #    - 'D:\\nltk_data'
    #    - 'E:\\nltk_data'
    #    - 'C:\\Users\\brynn\\Miniconda2\\nltk_data'
    #    - 'C:\\Users\\brynn\\Miniconda2\\lib\\nltk_data'
    #    - 'C:\\Users\\brynn\\AppData\\Roaming\\nltk_data'
    
    #####################
    # LOADING RESOURCES #
    #####################
    
    # load the cleaned data set
    data = load_data("test.tsv")
    # X contains feature vectors for each "fake news" article.
    # y contains the fake news classification from the BS Detector, which we
    # will use to analyze our clusters but not to create them.
    
    # The first line of the tsv is field headers.
    headers = np.append(data.X[0, :], [data.y[0]])
    num_fields = len(headers) - 1
    X, y = data.X[1:, :], data.y[1:]
    
    # title (1), text (2), and thread title (5)
    text_fields = [2] # [1,2,5]
    
    # author (0), site url (3), country (4)
    categorical_fields = [0] #[0,3,4] 
    
    # spam score (6), replies count (7), participants (8), likes (9), comments (10), shares (11)
    numerical_fields = [1] #[6,7,8,9,10,11] 
    
    ###################
    # PROCESSING TEXT #
    ###################
    
    feature_matrix, vectorizer, transformer = extract_feature_vectors(X, text_fields, numerical_fields)

    print X
    print feature_matrix
    
    #model = kprototypes.KPrototypes(n_clusters=2, init='Cao', verbose=2)
    #clusters = model.fit_predict(X, categorical=categorical_indices)
    #print clusters

            
if __name__ == "__main__" :
    main()
    
    
    
