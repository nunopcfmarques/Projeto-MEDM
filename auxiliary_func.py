from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, recall_score, make_scorer, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import recall_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# Não existe "scorer" de specificity na biblioteca, pelo que se cria
specificity = make_scorer(recall_score, pos_label=0)


def cv_fit(classifier, X, y, cv=10):
    # Função auxiliar que dado um classificador, X e y faz a "cross-validation" definida pelo argumento cv
    # Devolve as métricas
    acc = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')

    err = 1 - acc

    rec = cross_val_score(classifier, X, y, cv=cv, scoring='recall')

    pre = cross_val_score(classifier, X, y, cv=cv, scoring='precision')

    spe = cross_val_score(classifier, X, y, cv=cv, scoring=specificity)

    fpr = 1. - spe

    f1_score = cross_val_score(classifier, X, y, cv=cv, scoring='f1')

    return [np.mean(acc), np.mean(err), np.mean(rec), np.mean(pre), np.mean(fpr), np.mean(spe), np.mean(f1_score)]


def print_met(mets):
    # Função auxiliar que mostra as métricas

    print(str(mets[0]) + " & " + str(mets[3]) + " & " +
          str(mets[2]) + " & " + str(mets[6]) + "\n")

    print("Accuracy: " + str(mets[0]) + "\n" + "Error: " + str(mets[1]) + "\n" + "Recall: " + str(mets[2]) + "\n" +
          "Precision: " + str(mets[3]) + "\n" + "FPR: " + str(mets[4]) + "\n" + "Specifity: " + str(mets[5]) + "\n" + "f1_score: " + str(mets[6]) + "\n")


def plot_met(cm):
    # Função auxiliar que faz "plot" da matriz de confusão
    cm_disp = ConfusionMatrixDisplay(cm)

    cm_disp.plot(include_values=True, cmap='plasma',
                 ax=None, xticks_rotation='horizontal')

    plt.grid(False)
    plt.show()


def bayes(classifier, X, y):

    # NB com K Fold

    print("NB com K Fold \n")

    results = cv_fit(classifier, X, y)

    print_met(results)

    # NB com repeated K Fold

    print("NB com Repeated K Fold \n")

    results = cv_fit(classifier, X, y, cv=RepeatedKFold(
        n_splits=5, n_repeats=10, random_state=20))

    print_met(results)

    # NB com Stratified K Fold

    print("NB com Stratified K Fold \n")

    results = cv_fit(classifier, X, y, cv=StratifiedKFold())

    print_met(results)
