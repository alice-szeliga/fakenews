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

from sklearn.feature_extraction.text import TfidfTransformer

def extract_dictionary(field, X):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        field    -- int, index of field to extract a dictionary for
        X        -- ndarray of dimensions (n,d), data samples
    
    Returns
    --------------------
        field_dictionary -- dictionary, (key, value) pairs are (word, index)
        proccessed_text  -- list of lists, processed text fields for each sample
    """
    n,d = X.shape
    
    field_dictionary = {}
    processed_text = [[] for i in range(n)]
    index = 0
    
    stopwords = nltk.corpus.stopwords.words("english")
    
    for i in range(n):
        field_text = X[i, field]
        # tokenize
        tokens = nltk.word_tokenize(field_text)
        # remove stop words
        words = [w for w in tokens if not w in stopwords]
        
        # process the text to populate word_list
        for word in words:
            if word not in field_dictionary:
                field_dictionary[word] = index
                index += 1
        
        # store the words in processed_text
        processed_text[i] = words

    return field_dictionary, processed_text


def extract_feature_vectors(X, text_fields, field_dictionaries):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        X         -- ndarray of dimensions (n,d), data samples
        text_fields      -- dictionary, pairs are (field index, list of processed text for that field)
        field_dictionaries -- dictionary, pairs are (field index, dictionary for that field)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d), each feature is either
                          the original field value (for numbers) or a bag-of-words
                          representation (for text):
                          boolean (0,1) array indicating word presence in a string
    """
    # create appropriately sized matrix to hold feature vectors
    n,d = X.shape
    num_features = d
    for fd in field_dictionaries.itervalues():
        num_features += len(fd) - 1
    feature_matrix = np.empty((n, num_features), dtype=object)
    
    # Create a TF-IDF transformer to normalize data.
    transformer = TfidfTransformer(smooth_idf=False)
    
    # process each line to populate feature_matrix
    for i in range(n):
        field_feature_index = 0
        # loop over fields and expand textual ones to be expressed as bag-of-
        # words across multiple features
        for j in range(d):
            if j in text_fields:
                field_text = text_fields[j][i] # textual entry for that field
                field_dictionary = field_dictionaries[j]
                for word in field_text:
                    feature_index = field_feature_index + field_dictionary[word]
                    if feature_matrix[i, feature_index] == None:
                        feature_matrix[i, feature_index] = 1
                    else:
                        feature_matrix[i, feature_index] += 1
#                for word in field_dictionary:
#                    feature_index = field_feature_index + field_dictionary[word]
#                    if word in field_text:
#                        feature_matrix[i, feature_index] = 1
#                    else:
#                        feature_matrix[i, feature_index] = 0
                field_feature_index += len(field_dictionary)
            else:
                print X[i,j]
                feature_matrix[i, field_feature_index] = X[i,j]
                print feature_matrix[i, field_feature_index]
                field_feature_index += 1

    return feature_matrix

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
    
    long_text_fields = ["text"] #["title", "text", "thread_title"]
    categorical_fields = ["author"] #["author", "site_url"]
    categorical_indices = [0] #[0, 3]
    
    ###################
    # PROCESSING TEXT #
    ###################
    
    field_dictionaries = {}
    text_fields = {}
    for i in range(num_fields):
        if headers[i] in long_text_fields:
            field_dictionaries[i], text_fields[i] = extract_dictionary(i, X)
    
    feature_matrix = extract_feature_vectors(X, text_fields, field_dictionaries)
    print X
    print feature_matrix
    
    #model = kprototypes.KPrototypes(n_clusters=2, init='Cao', verbose=2)
    #clusters = model.fit_predict(X, categorical=categorical_indices)
    #print clusters
    
    ########
    # TEST #
    ########
#    data = load_data("test.tsv")
#    headers = np.append(data.X[0, :], [data.y[0]])
#    X, y = data.X[1:, :], data.y[1:]
#    
#    long_text_fields = ["author", "text"]
#
#    stopwords = nltk.corpus.stopwords.words("english")
#    
#    test_sample = X[0, :]
#    for i in range(len(test_sample)):
#        header = headers[i]
#        field_value = test_sample[i]
#        print "Field #", i, header, ":"
#        if header in long_text_fields:
#            # tokenization
#            tokens = nltk.word_tokenize(field_value)
#            # removing stop words
#            words = [w for w in tokens if not w in stopwords]
#            print words
#        else:
#            print field_value
            
if __name__ == "__main__" :
    main()
    
    
    
