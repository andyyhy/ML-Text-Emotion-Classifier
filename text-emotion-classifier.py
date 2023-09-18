"""EECS 445 - Winter 2022.
Project 1
"""
from audioop import maxpp
from math import gamma
from netrc import netrc
from tracemalloc import stop
from xxlimited import new
import pandas as pd
import numpy as np
import itertools
import string
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt
from helper import *
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
np.random.seed(445)
def extract_word(input_string):
    """Preprocess review into list of tokens.
    Convert input string to lowercase, replace punctuation with spaces, and split al
    ong whitespace.
    Return the resulting array.
    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]
    Input:
    input_string: text for a single review
    Returns:
    a list of words, extracted and preprocessed according to the directions
    above.
    """
    sentence = input_string
    punc = string.punctuation
    for ele in punc:
        sentence = sentence.replace(ele, " ")
    sentence = sentence.lower()
    wordList = sentence.split()
    return wordList

def extract_dictionary(df):
    """Map words to index.
    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).
    E.g., with input:
    | text | label | ... |
    | It was the best of times. | 1 | ... |
    | It was the blurst of times. | -1 | ... |
    The output should be a dictionary of indices ordered by first occurence inthe entire dataset:
    {
    it: 0,
    was: 1,
    the: 2,
    best: 3,
    of: 4,
    times: 5,
    blurst: 6
    }
    The index should be autoincrementing, starting at 0.
    Input:
    df: dataframe/output of load_data()
    Returns:
    a dictionary mapping words to an index
    """
    word_dict = {}
    counter = 0
    for text in df["text"]:
        wordList = extract_word(text)
        for ele in wordList:
            if ele not in word_dict:
                word_dict[ele] = counter
                counter+=1
    return word_dict

def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.
    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review. Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).
    Input:
    df: dataframe that has the text and labels
    word_dict: dictionary of words mapping to indices
    Returns:
    a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    counter = 0
    #iterate through each row of the data
    for text in df["text"]:
        #print(text)
        wordList = extract_word(text)
        for ele in wordList:
            if ele in word_dict:
                feature_matrix[counter][word_dict[ele]] = 1
        counter += 1
    return feature_matrix

def performance(y_true, y_pred, metric="accuracy", multi_class="raise"):
    """Calculate performance metrics.
    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.
    Input:
    y_true: (n,) array containing known labels
    y_pred: (n,) array containing predicted scores
    metric: string specifying the performance metric (default='accuracy'
    other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
    and 'specificity')
    Returns:
    the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if(metric == "auroc"):
        return (metrics.roc_auc_score(y_true, y_pred, multi_class= multi_class))
    elif(metric == "accuracy"):
        return(metrics.accuracy_score(y_true, y_pred))
    elif(metric == "f1_score"):
        return(metrics.f1_score(y_true, y_pred))
    elif(metric == "precision"):
        return(metrics.precision_score(y_true, y_pred))
    elif(metric == "sensitivity"):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        return(tp/(tp+fn))
    elif(metric == "specificity"):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        return(tn/(tn+fp))
    else:
        print("Error! undefined metric")

def cv_performance(clf, X, y, k=5, metric="accuracy", multi_class="raise"):
    """Split data into k folds and run cross-validation.
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
    clf: an instance of SVC()
    X: (n,d) array of feature vectors, where n is the number of examples
    and d is the number of features
    y: (n,) array of binary labels {1,-1}
    k: an int specifying the number of folds (default=5)
    metric: string specifying the performance metric (default='accuracy'
    other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
    and 'specificity')
    Returns:
    average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    # Put the performance of the model on each fold in the scores array
    scores = []
    skf = StratifiedKFold(n_splits=k)
    for train, test in skf.split(X, y):
        clf.fit(X[train], y[train])
        if(metric == "auroc"):
            predicted_labels = clf.decision_function(X[test])
            actual_labels = y[test]
            scores.append(performance(actual_labels, predicted_labels, metric, multi_class))
        else:
            predicted_labels = clf.predict(X[test])
            actual_labels = y[test]
            scores.append(performance(actual_labels, predicted_labels, metric))
    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True,
    multi_class = "raise"
    ):
    """Search for hyperparameters of linear SVM with best k-fold CV performance.
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
    X: (n,d) array of feature vectors, where n is the number of examples
    and d is the number of features
    project1.py Page 4
    y: (n,) array of binary labels {1,-1}
    k: int specifying the number of folds (default=5)
    metric: string specifying the performance metric (default='accuracy',
    other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
    and 'specificity')
    C_range: an array with C values to be searched over
    loss: string specifying the loss function used (default="hinge",
    other option of "squared_hinge")
    penalty: string specifying the penalty type used (default="l2",
    other option of "l1")
    dual: boolean specifying whether to use the dual formulation of the
    linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
    the parameter value for a linear-kernel SVM that maximizes the
    average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    max_permformance = 0
    max_C = 0
    for ele in C_range:
        if multi_class == "raise":
            clf = LinearSVC(penalty=penalty,loss = loss, dual = dual, class_weight="balanced", C=ele, random_state=445)
        else:
            clf = OneVsRestClassifier(LinearSVC(penalty=penalty,loss = loss, dual = dual, class_weight="balanced", C=ele, random_state=445))
            performance = cv_performance(clf, X, y, k=5, metric = metric, multi_class = multi_class)
        if(performance > max_permformance):
            max_permformance = performance
            max_C = ele
    return max_C

def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.
    Input:
    X: (n,d) array of feature vectors, where n is the number of examples
    and d is the number of features
    y: (n,) array of binary labels {1,-1}
    penalty: penalty to be forwarded to the LinearSVC constructor
    C_range: list of C values to train a classifier on
    loss: loss function to be forwarded to the LinearSVC constructor
    dual: whether to solve the dual or primal optimization problem, to be
    forwarded to the LinearSVC constructor
    Returns: None
    Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for ele in C_range:
        clf = LinearSVC(penalty=penalty,loss = loss, dual = dual, class_weight="balanced", C=ele, random_state=445)
        clf.fit(X, y)
        n0 = 0
        for theta in clf.coef_:
            for i in theta:
                if i !=0:
                    n0+=1
        norm0.append(n0)
    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()

def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters of quadratic SVM with best k-fold CV performance.
    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
    X: (n,d) array of feature vectors, where n is the number of examples
    and d is the number of features
    y: (n,) array of binary labels {1,-1}
    k: an int specifying the number of folds (default=5)
    metric: string specifying the performance metric (default='accuracy'
    other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
    and 'specificity')
    param_range: a (num_param, 2)-sized array containing the
    parameter values to search over. The first column should
    represent the values for C, and the second column should
    represent the values for r. Each row of this array thus
    represents a pair of parameters to be tried together.
    Returns:
    The parameter values for a quadratic-kernel SVM that maximize
    the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    max_performance = 0
    for c, r in param_range:
        clf = SVC(kernel="poly", degree=2, C=c, coef0=r, gamma="auto")
        performance = cv_performance(clf, X, y, k, metric)
        if performance > max_performance:
            best_C_val = c
            best_r_val = r
            max_performance = performance
    return best_C_val, best_r_val

def main():
    normal = 0
    if normal:
        X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(fname="data/dataset.csv")
        IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(dictionary_binary, fname="data/dataset.csv")
        print("Question 3(d): reporting dataset statistics:\n")
        #For 3a
        q3a = extract_word("¿BEST book ever! It\¿s great¿")
        print("The processed sentence is ", q3a)
        #For 3b
        print("d: ", len(dictionary_binary))
        #For 3c Average number of nonzero features
        sum_nonzero = []
        for row in X_train:
            sum_nonzero.append(np.sum(row))

        print("Average number of nonzero features: ", np.array(sum_nonzero).mean())
        #For 3c Word appearing in the most number of comments
        most_common_word = ""
        most_appearance = 0
        transposed = zip(*dictionary_binary)
        for key, value in dictionary_binary.items():
            sum = 0
            for row in X_train:
                sum += row[value]
            if sum > most_appearance:
                most_appearance = sum
                most_common_word = key
        print("Most common word: ", most_common_word)
        print("--------------------------------------------")
        #For 4.1(b) Best c and performance
        print("Question 4.1(b): Best C value and CV performance by metric")
        my_metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
        C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        max_C = 0
        for ele in my_metrics:
            print("metric: ", ele)
            max_C = select_param_linear(X_train, Y_train, 5, ele, C_range= C)
            clf = LinearSVC(random_state=445, C=max_C,loss="hinge", penalty="l2", dual=True)
            print("Best c: ", max_C)
            print("CV Score ", cv_performance(clf, X_train, Y_train, 5 , ele))
            print()
            #For 4.1(c)
        for ele in my_metrics:
            print("metric: ", ele)
            clf = LinearSVC(random_state=445, C=1,loss="hinge", penalty="l2", dual=True)
            print("Performance Score ", cv_performance(clf, X_train, Y_train, 5 , ele))
            print()

        #For 4.1(d)
        plot_weight(X_train, Y_train, penalty="l2",C_range=C,loss="hinge", dual=True)
        print("--------------------------------------------")
        print("Question 4.1e Most positive and Most negative Words")
        #For 4.1(e)
        clf = LinearSVC(random_state=445, C=0.1,loss="hinge", penalty="l2", dual=True)
        clf.fit(X_train, Y_train)
        coef = clf.coef_[0].argsort()
        most_pos_index = coef[:-6:-1]
        most_neg_index = coef[:5]
        most_pos_words = []
        most_neg_words = []
        for ele in most_pos_index:
            for key, value in dictionary_binary.items():
                if ele == value:
                    most_pos_words.append(key)
        for ele in most_neg_index:
            for key, value in dictionary_binary.items():
                if ele == value:
                    most_neg_words.append(key)
        for i in range(5):
            print(clf.coef_[0,most_pos_index[i]], most_pos_words[i])
        for i in range(5):
            print(clf.coef_[0,most_neg_index[i]], most_neg_words[i])
        #For 4.2a
        print("--------------------------------------------")
        print("4.2a Report the C value, the mean CV AUROC score, and the AUROC scoreon the test set.")
        C = [0.001, 0.01, 0.1, 1]
        max_C = 0
        print("metric: ", 'auroc')
        max_C = select_param_linear(X_train, Y_train, 5, "auroc", C_range= C,loss="squared_hinge", penalty="l1", dual=False)
        clf = LinearSVC(random_state=445, C=max_C,loss="squared_hinge", penalty="l1", dual=False)
        print("Best c: ", max_C)
        print("CV Score ", cv_performance(clf, X_train, Y_train, 5 , "auroc"))
        clf.fit(X_train, Y_train)
        predicted_labels = clf.decision_function(X_test)
        actual_labels = Y_test
        auroc = metrics.roc_auc_score(actual_labels, predicted_labels)
        print("Test Performance: ", auroc)
        print()
        #For 4.2b
        plot_weight(X_train, Y_train, penalty="l1",C_range=C,loss="squared_hinge", dual=False)

        print("--------------------------------------------")
        #For 4.3 Grid Search
        R = [0.01, 0.1, 1, 10, 100, 1000]
        C = [0.01, 0.1, 1, 10, 100, 1000]
        CR = []
        for c in C:
            for r in R:
                CR.append([c, r])
        [max_C, max_R] = select_param_quadratic(X_train, Y_train, 5, "auroc", CR)
        print("Quadratic SVM with grid search and auroc metric:")
        print("Best c: ", max_C, " Best coeff: ", max_R)
        clf = SVC(kernel="poly", degree=2, C=max_C, coef0=max_R, gamma="auto", class_weight="balanced")
        clf.fit(X_train, Y_train)
        predicted_labels = clf.decision_function(X_test)
        actual_labels = Y_test
        auroc = metrics.roc_auc_score(actual_labels, predicted_labels)
        print("Test Performance: ", auroc)
        print()
        #For 4.3 Random Search
        CR = []
        for i in range(25):
            c = np.random.uniform(-5, 5)
            r = np.random.uniform(-5, 5)
            CR.append([10**c, 10**r])
        [max_C, max_R] = select_param_quadratic(X_train, Y_train, 5, "auroc", CR)
        print("Quadratic SVM with random search and auroc metric:")
        print("Best c: ", max_C, " Best coeff: ", max_R)
        clf = SVC(kernel="poly", degree=2, C=max_C, coef0=max_R, gamma="auto", class_weight="balanced")
        clf.fit(X_train, Y_train)
        predicted_labels = clf.decision_function(X_test)
        actual_labels = Y_test
        auroc = metrics.roc_auc_score(actual_labels, predicted_labels)
        print("Test Performance: ", auroc)

        #For 5.1
        print("Question 5.1: Linear SVM with imbalanced class weights")
        my_metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
        C = 0.01
        clf = LinearSVC(random_state=445, C=C,loss="hinge", penalty="l2", dual=True, class_weight={-1:1, 1:10})
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        Y_pred_auroc = clf.decision_function(X_test)
        for ele in my_metrics:
            if ele == "auroc":
                print("Test Performance on metric ", ele, ": ", performance(Y_test, Y_pred_auroc, ele))
            else:
                print("Test Performance on metric ", ele, ": ", performance(Y_test, Y_pred, ele))
        print("--------------------------------------------")
        #For 5.2
        print("Question 5.2: Linear SVM on an imbalanced data set")
        my_metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
        C = 0.01
        clf = LinearSVC(random_state=445, C=C,loss="hinge", penalty="l2", dual=True, class_weight={-1:1, 1:1})
        clf.fit(IMB_features, IMB_labels)
        Y_pred = clf.predict(IMB_test_features)
        Y_pred_auroc = clf.decision_function(IMB_test_features)
        for ele in my_metrics:
            if ele == "auroc":
                print("Test Performance on metric ", ele, ": ", performance(IMB_test_labels, Y_pred_auroc, ele))
        else:
                print("Test Performance on metric ", ele, ": ", performance(IMB_test_labels, Y_pred, ele))

        print("--------------------------------------------")

        #For 5.3a
        print("Question 5.3: Choosing appropriate class weights")
        W_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        best_wn = 0
        best_wp = 0
        max_performance = 0
        for wn in W_range:
            for wp in W_range:
                clf = LinearSVC(random_state=445, C=0.01,loss="hinge", penalty="l2", dual=True, class_weight={-1:wn, 1:wp})
                perf = cv_performance(clf, IMB_features, IMB_labels, metric="auroc")
                print(wn, " ", wp, " ", perf)
                if perf > max_performance:
                    max_performance = perf
                    best_wn = wn
                    best_wp = wp
        print("For AUROC")
        print("best Wn: ", best_wn)
        print("best wp: ", best_wp)
        print("best performance: ", max_performance)

        print()
        #For 5.3b
        my_metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
        C = 0.01
        clf = LinearSVC(random_state=445, C=C,loss="hinge", penalty="l2", dual=True,
        class_weight={-1:10, 1:7})
        clf.fit(IMB_features, IMB_labels)
        Y_pred = clf.predict(IMB_test_features)
        Y_pred_auroc = clf.decision_function(IMB_test_features)
        for ele in my_metrics:
            if ele == "auroc":
                print("Test Performance on metric ", ele, ": ", performance(IMB_test_labels, Y_pred_auroc, ele))
            else:
                print("Test Performance on metric ", ele, ": ", performance(IMB_test_labels, Y_pred, ele))
        #For 5.4
        C = 0.01
        clf_normal = LinearSVC(random_state=445, C=C,loss="hinge", penalty="l2", dual=True, class_weight={-1:1, 1:1})
        clf_custom = LinearSVC(random_state=445, C=C,loss="hinge", penalty="l2", dual=True, class_weight={-1:10, 1:7})
        clf_normal.fit(IMB_features, IMB_labels)
        clf_custom.fit(IMB_features, IMB_labels)
        axes = plt.gca()
        metrics.plot_roc_curve(clf_normal, IMB_test_features, IMB_test_labels, ax=axes, name = 'Balanced $W_n:W_p = 1:1$')
        metrics.plot_roc_curve(clf_custom, IMB_test_features, IMB_test_labels, ax=axes, name = 'Custom $W_n:W_p = 10:7$')
        plt.savefig("ROC Plot")
        plt.close()
    else:
        #Feature engineering
        (multiclass_features,
        multiclass_labels,
        multiclass_dictionary) = get_multiclass_training_data()
        stopwords = ["i", "a", "an", "are", "as", "at", "be", "by", "for", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "who", "will", "with", "you", "your", "we"]
        heldout_features = get_heldout_reviews(multiclass_dictionary)

        #Count the frequency of each word
        #Get rid of words in Stopwords
        cols = []
        for word in stopwords:
            if word in multiclass_dictionary:
                cols.append(multiclass_dictionary[word])

        new_multiclass_features = np.delete(multiclass_features, cols, 1)
        frequency = np.count_nonzero(np.array(multiclass_features), axis=0)
        #Get rid of words that appear less than 3 times
        new_multiclass_features = multiclass_features[:, frequency > 2]
        C_range = [0.001 ,0.01, 0.1, 1, 10, 100, 1000]
        r_range = [0.001 ,0.01, 0.1, 1, 10, 100, 1000]
        CR = []
        for i in range(25):
            c = np.random.uniform(-3, 3)
            r = np.random.uniform(-3, 3)
            CR.append([10**c, 10**r])
        W_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #Testing Linear Kernel with l1 penalty
        print(multiclass_features.shape)
        print(new_multiclass_features.shape)
        print(multiclass_labels.shape)
        print("Linear Kernel with l2")
        best_c = select_param_linear(new_multiclass_features, multiclass_labels, 5, "accuracy", C_range= C_range, multi_class="ovr")
        print("Best C: ", best_c)
        best_wn = 0
        best_wp = 0
        max_performance = 0
        for wn in W_range:
            for wp in W_range:
                clf = LinearSVC(random_state=445, C=best_c,loss="hinge", penalty="l2", dual=True, class_weight={-1:wn, 1:wp})
                perf = cv_performance(clf,new_multiclass_features, multiclass_labels, metric="accuracy", multi_class="ovr")
                if perf > max_performance:
                    max_performance = perf
                    best_wn = wn
                    best_wp = wp
        print("Best W_n: ", best_wn)
        print("Best W_p: ", best_wp)
        clf = LinearSVC(random_state=445, C=best_c,loss="hinge", penalty="l2", dual=True, class_weight={-1:best_wn, 1:best_wp})
        perf = cv_performance(clf, new_multiclass_features, multiclass_labels, 5, "accuracy", multi_class="ovr")
        print("Performance: ", perf)
        print()
        print("Linear Kernel with l1")
        best_c = select_param_linear(new_multiclass_features, multiclass_labels, 5,
        "accuracy", C_range= C_range, multi_class="ovr", penalty="l1", loss="squared_hinge",dual=False)
        print("Best C: ", best_c)
        best_wn = 0
        best_wp = 0
        max_performance = 0
        for wn in W_range:
            for wp in W_range:
                clf = LinearSVC(random_state=445, C=best_c,loss="squared_hinge", penalty="l1", dual=False, class_weight={-1:wn, 1:wp})
                perf = cv_performance(clf,new_multiclass_features, multiclass_labels, metric="accuracy", multi_class="ovr")
                if perf > max_performance:
                    max_performance = perf
                    best_wn = wn
                    best_wp = wp
        print("Best W_n: ", best_wn)
        print("Best W_p: ", best_wp)
        clf = LinearSVC(random_state=445, C=best_c,loss="squared_hinge", penalty="l1", dual=False, class_weight={-1:best_wn, 1:best_wp})
        perf = cv_performance(clf, new_multiclass_features, multiclass_labels, 5, "accuracy", multi_class="ovr")
        print("Performance: ", perf)
        print()
        print("Quadratic Kernel")
        [best_c, best_r] = select_param_quadratic(new_multiclass_features, multiclass_labels, 5, "accuracy", param_range=CR)
        print("Best C: ", best_c)
        print("Best r: ", best_r)
        best_wn = 0
        best_wp = 0
        max_performance = 0
        for wn in W_range:
            for wp in W_range:
                clf = SVC(kernel="poly", degree=2, C=best_c, coef0=best_r, gamma="auto", class_weight={-1:wn, 1:wp})
                perf = cv_performance(clf,new_multiclass_features, multiclass_labels, metric="accuracy", multi_class="ovr")
                if perf > max_performance:
                    max_performance = perf
                    best_wn = wn
                    best_wp = wp
        print("Best W_n: ", best_wn)
        print("Best W_p: ", best_wp)
        clf = SVC(kernel="poly", degree=2, C=best_c, coef0=best_r, gamma="auto", class_weight={-1:best_wn, 1:best_wp})
        perf = cv_performance(clf, new_multiclass_features, multiclass_labels, 5, "accuracy", multi_class="ovr")
        print("Performance: ", perf)
        #Final
        clf = LinearSVC(random_state=445, C=1,loss="squared_hinge", penalty="l1", dual=False, class_weight={-1:1, 1:1})
        clf.fit(multiclass_features, multiclass_labels)
        my_labels = clf.predict(heldout_features)
        generate_challenge_labels(my_labels, "andyyhy")
    if __name__ == "__main__":
        main()