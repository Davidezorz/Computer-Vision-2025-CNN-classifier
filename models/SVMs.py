import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import manhattan_distances
import seaborn as sns
import matplotlib.pyplot as plt






class MulticlassECOC_SVM:
    """
    Implements a Multiclass SVM using the ECOC (Error Correcting Output Codes) 
    approach.
    Reference: Dietterich & Bakiri (1995).
    """

    def __init__(self, kernel='linear', L=15, C=1.0):
        self.kernel = kernel
        self.C = C
        self.L = L
        self.SVMs = []
        self.code_book = None


    def generateCodebook(self, n_classes, n_iterations=500):
        size = (n_classes, self.L)
        code_book = np.random.randint(0, 2, size=size)
        
        for _ in range(n_iterations):
            # Find worst pair of ROWS
            row_dists = manhattan_distances(code_book, code_book)               # Manhattan dist on binary data == Hamming distance
            np.fill_diagonal(row_dists, np.inf)                                 # Avoid self comparison

            row_min_idx = np.argmin(row_dists)                                  # argmin gives index in flattened array,
            r1, r2 = np.unravel_index(row_min_idx, row_dists.shape)             # converts back to (row, col) indexes
            
            # Find extreme pair of COLUMNS
            col_dists = manhattan_distances(code_book.T, code_book.T)           # code_book.T to calculate distances between columns
            np.fill_diagonal(col_dists, 0)                                      # Ignore diagonal for now

            optimal_dist = n_classes / 2.0                                      # compute the mean value
            deviation = np.abs(col_dists - optimal_dist)                        # compute distances w.r.t. the mean value
            np.fill_diagonal(deviation, -1)                                     # Avoid self comparison
            c1, c2 = np.unravel_index(np.argmax(deviation), deviation.shape)    # Find indices (c1, c2) of the most extreme columns
            
            # flip bits
            code_book[r2, c1] = 1 - code_book[r2, c1]                           # Flip the bits for r2 
            code_book[r2, c2] = 1 - code_book[r2, c2]                           # at columns c1 and c2
            
        return code_book


    def fit(self, X, y):
        self.classes = np.unique(y)                                             # extract the labels  
        n_classes = len(self.classes)   
        self.code_book = self.generateCodebook(n_classes)                       # compute the codebook
        
        for column in self.code_book.T:
            y_new = column[y]                                                   # compute the new labes

            svm = SVC(kernel=self.kernel, C=self.C)                             # instantiating SVM
            svm.fit(X, y_new)                                                   # fit SVM with new labes
            self.SVMs.append(svm)


    def predict(self, X):
        code_preds = np.zeros((X.shape[0], self.L))
        for i, svm in enumerate(self.SVMs):                                     # compute the predictions
            code_preds[:, i] = svm.predict(X)
        
        distances = manhattan_distances(code_preds, self.code_book)             # manhattan_distances computes sum(|x - y|)
        closest_class_indices = np.argmin(distances, axis=1)                    # which is Hamming distance for binary data
        y_pred = self.classes[closest_class_indices]
        return y_pred
        

    def plotCodebook(self, classes=None, show=True, save_path=None):
        """ Plots the ECOC codebook (Matrix of Codewords) as a heatmap.
        Visualizes the binary codes assigned to each class. """
        h, w = self.code_book.shape
        plt.figure(figsize=(max(w/2, 8), max(h/2, 6)))
        
        classes = classes if classes != None else [str(i) for i in range(h)]
        
        sns.heatmap(self.code_book, annot=True, fmt='d', cmap='binary', 
                    cbar=False,
                    xticklabels=[f"Bit {i}" for i in range(w)],
                    yticklabels=classes,
                    linewidths=1, linecolor='lightgray')
        
        plt.ylabel('Class Label')
        plt.xlabel('SVM Classifier Index (Code Bits)')
        plt.title(f'ECOC Codebook ({w} bits)')
        
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        _ = plt.show() if show else plt.close()















class MulticlassDAG_SVM:
    """
    Implements a Multiclass SVM using the DAG (Directed Acyclic Graph) approach.
    It builds K(K-1)/2 classifiers (One-vs-One).
    During prediction, it uses a graph traversal to eliminate classes one by one.
    """
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.classifiers = {} # Dictionary to store binary classifiers (i, j)
        self.classes = []


    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        print(f"Training {n_classes * (n_classes - 1) // 2} binary SVMs for DAG...")
        
        for i in range(n_classes):                                              # Train One-vs-One classifiers
            for j in range(i + 1, n_classes):
                c1, c2 = self.classes[i], self.classes[j]
                
                mask = np.logical_or(y == c1, y == c2)                          # Select data only for class i and class j
                X_pair = X[mask]
                y_pair = y[mask]
                
                clf = SVC(kernel=self.kernel, C=self.C)                         # Train binary SVM
                clf.fit(X_pair, y_pair)
                
                self.classifiers[(c1, c2)] = clf                                # Store classifier indexed by tuple (min, max)
    

    def predict(self, X):
        predictions = []
        for sample in X:
            
            candidate_classes = list(self.classes)                              # Start with a list of all possible classes          
            sample = sample.reshape(1, -1)                                      # DAG traversal: Eliminate one candidate at a time
            
            while len(candidate_classes) > 1:
                c1 = candidate_classes[0]                                       # We compare the first and last
                c2 = candidate_classes[-1]                                      # candidate in the list
                
                key = tuple(sorted((c1, c2)))                                   # Retrieve the classifier for this pair
                clf = self.classifiers[key]                                     # We always stored it as (min, max)
                
                pred = clf.predict(sample)[0]
                
                if pred == c1:                                                  # If prediction is c1, c2 is not 
                    candidate_classes.pop(-1)                                   # the class -> remove c2
                else:                                                           # Otherwise
                    candidate_classes.pop(0)                                    # -> remove c1
            
            predictions.append(candidate_classes[0])
        return np.array(predictions)

