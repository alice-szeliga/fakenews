"""
Created on Sun Apr 09 21:39:20 2017
@title: classifying fake news
@author: Brynn Arborico and Alice Szeliga
"""
# python libraries
import collections

# numpy libraries
import numpy as np

from kmodes import kprototypes


# natural language toolkit
# http://www.nltk.org/
import nltk

# libraries specific to project
from util import *
from cluster import *

from sklearn import cluster
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model.tests.test_ridge import ind

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

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def extract_feature_vectors(X, text_fields, numerical_fields, max_features):
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
    n,d = X.shape
    vectorizers = {}
    transformers = {}

    for i in range(d):
        vectorizers[i] = CountVectorizer(stop_words='english', lowercase=False, min_df=1, max_df = 0.8, max_features=max_features)
        transformers[i] = TfidfTransformer(use_idf=True, smooth_idf=False)
    
    feature_matrix = np.empty((n, 0), dtype=object)
    for field in range(d):
        if field in text_fields:
            field_text = X[:, field]
            counts = vectorizers[field].fit_transform(field_text)
            tfidf = transformers[field].fit_transform(counts)
            feature_matrix = np.concatenate((feature_matrix, tfidf.toarray()), axis=1)
            
        else:
            field_values = X[:, field]
            if field in numerical_fields:
                field_values = field_values.astype(np.float)
            feature_matrix = np.concatenate((feature_matrix, field_values.reshape(n, 1)), axis=1)
            
    return feature_matrix, vectorizers, transformers


def calculate_purity(clusters, true_labels, num_clusters, weighted=False):
    """
    Computes the proportion of the dominant class in each cluster and averages
    this across all clusters.
    
    Parameters
    --------------------
        clusters    -- output of kprototypes fit_predict function, the set of
                       clusters
        true_labels -- (n,) array of known labels for the features
        weighted    -- bool, whether to weight the purities of each cluster
                       when averaging
    
    Returns
    --------------------
        overall_purity -- a float between 0 and 1 representing the averaged
                          purity of the clusters
    """
    # dictionary {label, index} of all the unique label
    # with indices associated to make a list of unique counts
    unique_labels = {}
    ind = 0
    for label in true_labels:
        if label not in unique_labels:
            unique_labels[label] = ind
            ind += 1
            
    num_unique_labels = len(unique_labels)
    num_points = len(true_labels)

    # count the number of points with each label in each cluster
    cluster_label_counts = [[0 for i in range(num_unique_labels)] for i in range(num_clusters)]
    for i in range(num_points):
        cluster = clusters.labels_[i]
        label = true_labels[i]
        label_index = unique_labels[label]
        cluster_label_counts[cluster][label_index] += 1
    
    overall_purity = 0
    for label_count in cluster_label_counts:
        dominant_class_count = max(label_count)
        total_count = sum(label_count)
        purity = dominant_class_count / float(total_count)
        
        if weighted:
            weight = total_count / float(num_points)
        else:
            weight = 1 / float(num_clusters)
        overall_purity += weight * purity

    return overall_purity
    
def calculate_best_num_clusters(cluster_penalties):
    """
    Uses the elbow method to find the best number of clusters. Implemented
    according to http://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve.
    
    Parameters
    --------------------
        cluster_penalties -- dictionary of {num_clusters, penalty}, either
                             costs or purities
    
    Returns
    --------------------
        best_num_clusters -- the ideal number of clusters according to the
                             elbow method
    """
    num_clusters_range = cluster_penalties.keys()
    
    min_num_clusters = min(num_clusters_range)
    a = np.array([min_num_clusters, cluster_penalties[min_num_clusters]])
    max_num_clusters = max(num_clusters_range)
    b = np.array([max_num_clusters, cluster_penalties[max_num_clusters]])
    line = a - b
    unit_line = line / np.linalg.norm(line)
    
    best_num_clusters = 0
    max_d = -1
    for num_clusters in num_clusters_range:
        p = np.array([num_clusters, cluster_penalties[num_clusters]])
        d = np.linalg.norm(p - np.dot(p, unit_line) * unit_line)
        if d > max_d:
            max_d = d
            best_num_clusters = num_clusters
    
    return best_num_clusters


def main():

    #####################
    # LOADING RESOURCES #
    #####################
    
    # load the cleaned data set

    # X contains feature vectors for each "fake news" article.
    # y contains the fake news classification from the BS Detector, which we
    # will use to analyze our clusters but not to create them.
    
    # The first line of the tsv is field headers.

    # only run this once. afterwards, can load in X and y easily
    #"""
    print "start"
    data = load_data2("fake_subset_no_blanks_16.txt")
    print "done loading data"
    
    headers = np.append(data.X[0, :], [data.y[0]])
    X, y = data.X[1:, :], data.y[1:]
    # can't save on knuth
    #np.save("labels", y)
    #np.save("values", X)
    #np.save("headers", headers)
    print "done importing data"
    """
    y = np.load("labels.npy")
    X = np.load("values.npy")
    headers = np.load("headers.npy")
    print "done loading from files"
    """
    n, d = X.shape
    
    # Defining the text vs. categorical vs numerical fields for our d
    # title (1), text (2), and thread title (5)
    text_fields = [1,2,5]
    # author (0), site url (3), country (4)
    categorical_fields = [0,3,4] 
    # spam score (6), replies count (7), participants (8), likes (9), comments (10), shares (11)
    numerical_fields = [6,7,8,9,10,11] 
    
    ###################
    # PROCESSING TEXT #
    ###################
    max_features = 100
    num_fields = len(headers) - 1
    # for k means, we will treat categorical fields like tex
    kmeans_text_fields = text_fields + categorical_fields
    kmeans_text_fields.sort()
    print "building feature matrix"
    
    feature_matrix, vectorizers, transformers = extract_feature_vectors(X, text_fields, numerical_fields, max_features = max_features)
    print "done extracting feature vectors for k prototypes"
    kmeans_feature_matrix, vectorizer, transformer = extract_feature_vectors(X, kmeans_text_fields, numerical_fields, max_features = max_features)
    print "done extracting feature vectors for k means"

    # indices are shifted in the new features matrix because text columns get more features
    # we calculate the new indices for the original categorical indices
    categorical_indices_dict = {}
    field_index = 0
    for field in range(d):
        if field in text_fields:
            vectorizer = vectorizers[field]
            words = vectorizer.vocabulary_
            field_index += len(words)
        else:
            if field in categorical_fields:
                categorical_indices_dict[field] = field_index
            field_index += 1
    categorical_indices = categorical_indices_dict.values()
    
    ###################
    # HYPERPARAMETERS #
    ###################
    # use the elbow method to determine the number of clusters
    """
    num_clusters_range = range(2, 8)
   
    print "*********** K MEANS ***********"
    k_means_cluster_sets = {}
    k_means_cluster_impurities = {}
    k_means_cluster_costs = {}
    print "beginning elbow method to determine best # clusters"
    for i in num_clusters_range:
        print "  current number clusters is: ", i
        try:
            kmeans_model = cluster.KMeans(n_clusters = i)
            print "fitting"
            kmeans_model.fit_predict(kmeans_feature_matrix)
            k_means_cluster_sets[i] = kmeans_model
            # (same as 1 - purity since purity is a percentage)
            k_means_cluster_impurities[i] = 1 - calculate_purity(kmeans_model, y, i)
            k_means_cluster_costs[i] = kmeans_model.inertia_
            print k_means_cluster_costs
            print kmeans_model.inertia_
            print "for ", i, " clusters we have impurity of ", k_means_cluster_impurities[i], " and a cost of ", k_means_cluster_costs[i]
        except:
            print "    we got an error!!"
            print "    ", i, "clusters is probably too many clusters"
            break
    
    # see what we get by minimizing 
    best_num_clusters_by_costs = calculate_best_num_clusters(k_means_cluster_costs)
    best_num_cluster_by_impurities = calculate_best_num_clusters(k_means_cluster_impurities)
    print "FOR K MEANS"
    print "The best number of clusters (minimizing cost) is ", best_num_clusters_by_costs
    print "The best number of clusters (minimizing impurity) is ", best_num_cluster_by_impurities
    
    print "*********** K PROTOTYPES ***********"
    cluster_sets = {}
    cluster_impurities = {}
    cluster_costs = {}
    print "beginning elbow method to determine best # clusters"
    for i in num_clusters_range:
        print "  current number clusters is: ", i
        try:
            model = kprototypes.KPrototypes(n_clusters=i, init='Cao')
            print "fitting"
            model.fit_predict(feature_matrix, categorical=categorical_indices)
            cluster_sets[i] = model
            # (same as 1 - purity since purity is a percentage)
            cluster_impurities[i] = 1 - calculate_purity(model, y, i)
            cluster_costs[i] = model.cost_
            print "for ", i, " clusters we have impurity of ", cluster_impurities[i], " and a cost of ", cluster_costs[i]
        except:
            raise
            print "    we got an error!!"
            print "    ", i, "clusters is probably too many clusters"
            break
    
    # see what we get by minimizing 
    best_num_clusters_by_costs = calculate_best_num_clusters(cluster_costs)
    best_num_cluster_by_impurities = calculate_best_num_clusters(cluster_impurities)
    print "FOR K PROTOTYPES"
    print "The best number of clusters (minimizing cost) is ", best_num_clusters_by_costs
    print "The best number of clusters (minimizing impurity) is ", best_num_cluster_by_impurities
     
    ## looking at different values of gamma
    gamma_range = [0.01, 0.1, 1, 10, 100]
    clusters = 4
    cluster_sets = {}
    cluster_impurities = {}
    cluster_costs = {}
    print "examining different gamma values for k prototypes"
    for i in gamma_range:
        print "  current gamma value is: ", i
        try:
            model = kprototypes.KPrototypes(n_clusters=clusters, init='Cao', gamma = i, verbose=2)
            print "fitting"
            model.fit_predict(feature_matrix, categorical=categorical_indices)
            cluster_sets[i] = model
            # (same as 1 - purity since purity is a percentage)
            cluster_impurities[i] = 1 - calculate_purity(model, y, clusters)
            cluster_costs[i] = model.cost_
            print "for gamma = ", i, " we have impurity of ", cluster_impurities[i], " and a cost of ", cluster_costs[i]
        except:
            raise
            print "    we got an error!!"
            break
    
    # see what we get by minimizing 
    best_num_clusters_by_costs = calculate_best_num_clusters(cluster_costs)
    best_num_cluster_by_impurities = calculate_best_num_clusters(cluster_impurities)
    print "FOR K PROTOTYPES"
    print "The best number of clusters (minimizing cost) is ", best_num_clusters_by_costs
    print "The best number of clusters (minimizing impurity) is ", best_num_cluster_by_impurities
    """
    
    ###################
    #    ANALYSIS     #
    ###################
    num_clusters = 4
    g = 10
    # These are our final models
    k_proto_model = kprototypes.KPrototypes(n_clusters = num_clusters, init='Cao', gamma = g)##, verbose=2)
    print "fitting"
    k_proto_labels = k_proto_model.fit_predict(feature_matrix, categorical=categorical_indices)

    k_means_model = cluster.KMeans(n_clusters = num_clusters)
    print "fitting"
    k_means_labels = k_means_model.fit_predict(kmeans_feature_matrix)
    
    
    # get values for all n clusters
    labels = np.unique(k_proto_labels)
    k_proto_ind = [[] for x in range(num_clusters)]
    k_means_ind = [[] for x in range(num_clusters)]
    for i in range(n):
        # find which cluster this example maps to for k prototypes
        j = k_proto_labels[i] - 1
        k_proto_ind[j].append(i)
        # find which cluster this example maps to for k means
        j = k_means_labels[i] - 1
        k_means_ind[j].append(i)

    k_proto_counts = [len(x) for x in k_proto_ind]
    k_means_counts = [len(x) for x in k_means_ind]

    print "k prototypes"
    print "The cluster sizes are ", k_proto_counts
    print "k means"
    print "The cluster sizes are",  k_means_counts
        
    # getting actual rows
    k_proto_clusters = {}
    k_means_clusters = {}
    k_proto_y = {}
    k_means_y = {}
    for i in range(num_clusters):
        k_proto_clusters[i] = X[k_proto_ind[i], :]
        k_means_clusters[i] = X[k_means_ind[i], :]
        k_proto_y[i] = y[k_proto_ind[i]]
        k_means_y[i] = y[k_proto_ind[i]]
    
    k_proto_y_labels = {}
    k_means_y_labels = {}
    y_labels = np.unique(y)
    for i in range(num_clusters):
        #for label in y_labels:
        print k_proto_y[i]
        print [np.count_nonzero(k_proto_y[i] == label) for label in y_labels]
        k_proto_y_labels[i][label] = [np.count_nonzero(k_proto_y[i] == label) for label in y_labels]
        k_means_y_labels[i][label] = [np.count_nonzero(k_means_y[i] == label) for label in y_labels]
    print k_proto_y_labels[1]
    """
    for i in range(num_clusters):
        print "k prototypes"
        print "  for cluster ", i, " the labels are "
        kp_unique, kp_counts = np.unique(k_proto_y_labels[i], return_counts = True)
        print "  ", dict(zip(kp_unique, kp_counts))
    for i in range(num_clusters):
        print "k means"
        print "  for cluster ", i, " the labels are "
        km_unique, km_counts = np.unique(k_means_y_labels[i], return_counts = True)
        print "  ", dict(zip(km_unique, km_counts))
    """


    
    return

    
if __name__ == "__main__" :
    main()
    
    
    
