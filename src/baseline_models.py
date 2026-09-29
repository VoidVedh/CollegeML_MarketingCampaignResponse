# Baseline classification architectures: Logistic Regression & KNN
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def get_baseline_models():
    return {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier()}
