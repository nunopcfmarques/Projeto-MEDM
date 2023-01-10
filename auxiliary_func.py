import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix, recall_score, make_scorer, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

# Não existe "scorer" de specificity na biblioteca, pelo que se cria
specificity = make_scorer(recall_score, pos_label=0)


def get_met(y_test, y_pred):
    # Função auxiliar que dado o vetor real e o vetor previsto calcula as métricas
    # Devolve as métricas

    cm = confusion_matrix(y_test, y_pred)

    TN, FP, FN, TP = cm.ravel()

    N = TP + FN + FP + TN

    acc = (TP + TN)/N

    err = (FP + FN)/N

    rec = TP/(TP + FN)

    pre = TP/(TP + FP)

    fpr = FP/(FP + TN)

    spe = 1. - fpr

    f1_score = (2. * pre * rec)/(pre + rec)

    return [acc, err, rec, pre, fpr, spe, f1_score, cm]


def fit(classifier, X_train, X_test, y_train, y_test):
    # Função auxiliar que dado um classificador e o seu conjunto de treino e teste faz "fit"
    # Devolve as métricas

    clf = classifier.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return get_met(y_test, y_pred)


# (NOTA: define-se uma "seed" para o hold_out para resultados consistentes)
def hold_out(X, y, test_size=0.20):
    # Função auxiliar que dado X e y cria conjuntos treino e teste conforme uma partição aleatória dos dados.
    # Devolve o conjunto de treino e teste

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=20)

    return [X_train, X_test, y_train, y_test]


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
