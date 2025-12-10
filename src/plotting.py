from matplotlib import pyplot as plot
import sklearn.tree as tree
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def plot_decision_tree(clf : tree.DecisionTreeClassifier, activity_names:list):
    '''
        Plot the decision tree using matplotlib.
        Parameters:
            tree (tree.DecisionTreeClassifier): The decision tree classifier to plot.
    '''
    plot.figure(figsize=(30,20))
    tree_plot = tree.plot_tree(clf, filled=True, feature_names=activity_names, rounded=True)
    plot.savefig("docs/media/decision_tree.png")

def plot_confusion_matrix(true_labels, predicted_labels):
    '''
        Plot the confusion matrix using matplotlib.
        Parameters:
            true_labels: The true labels of the test set.
            predicted_labels: The predicted labels from the model.
    '''
    cm_display = ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels, cmap=plot.cm.Blues)
    cm_display.figure_.savefig("docs/media/confusion_matrix.png")

def compute_all_metrics(true_labels, predicted_labels):
    '''
        Compute and print all evaluation metrics.
        Parameters:
            true_labels: The true labels of the test set.
            predicted_labels: The predicted labels from the model.
        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1-measure.
    '''
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Measure: {f1*100:.2f}%")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_measure": f1
    }   