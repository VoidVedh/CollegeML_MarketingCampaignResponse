# Decision Tree and Gaussian Naive Bayes model setups
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def get_tree_nb_models():
    return {'Decision Tree': DecisionTreeClassifier(), 'Naive Bayes': GaussianNB()}
