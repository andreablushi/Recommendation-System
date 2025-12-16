from matplotlib import pyplot as plot
import matplotlib.patches as mpatches
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.types import BooleanCondition, Path

def plot_decision_tree(clf : DecisionTreeClassifier, activity_names:list, save:bool=False, path:str="docs/media/decision_tree.png"):
    """
    Plot the decision tree with confidence levels (probability of positive class) shown.
    
    Parameters:
        clf: DecisionTreeClassifier
        activity_names: List of feature names
        save: Whether to save the plot as PNG
    """
    # Use plot_tree with proportion=True to show probabilities, and custom labels
    plot.figure(figsize=(30, 20))
    tree.plot_tree(
        clf,
        feature_names=activity_names,
        filled=True,
        rounded=True,
        proportion=True,
        label='all',  
    )

    # Optionally save the figure
    if save:
        plot.savefig(path, bbox_inches='tight')
    plot.show()

def plot_recommendation_on_tree(clf: DecisionTreeClassifier, 
                                activity_names: list, 
                                prefix_trace_features: set, 
                                recommended_conditions: set, 
                                save: bool = False, 
                                path: str = "docs/media/recommendation_tree.png"):
    """
    Plot the decision tree with recommendations highlighted.
    
    Parameters:
        clf: DecisionTreeClassifier
        activity_names: List of feature names
        prefix_trace_features: Set of activity names in the prefix trace
        recommended_conditions: Set of (activity_name, value) tuples for recommendations
        save: Whether to save the plot as PNG
        path: Path to save the plot
    """
    # Create figure
    fig = plot.figure(figsize=(30, 20))
    ax = plot.gca()

    # Plot the basic decision tree
    tree_plot = tree.plot_tree(
        clf,
        feature_names=activity_names,
        filled=True,
        rounded=True,
        proportion=True,
        label='all',
    )
    
    # For each node in the tree
    for i, node in enumerate(tree_plot):
        if hasattr(node, 'get_text'):
            # Get node text and bbox
            node_text = node.get_text()
            bbox = node.get_bbox_patch()
            if bbox is None:
                continue
            
            # Check if the node corresponds to a feature in the prefix trace
            is_in_prefix = any(feature in node_text for feature in prefix_trace_features)
            
            # Check if node contains a recommended condition
            recommendation_value = None
            for feat, val in recommended_conditions:
                if feat in node_text:
                    recommendation_value = True
                    # Mark the
                    if val:
                        recommendation_value = True
                    else:
                        recommendation_value = False
                    break
            
            if is_in_prefix:
                # Node is in prefix trace: thick blue border
                bbox.set_edgecolor('blue')
                bbox.set_linewidth(6)
                bbox.set_linestyle('-')

            # If a recommendation exists for this node
            elif recommendation_value != None:
                if recommendation_value:
                    # Node is recommended as true: thick dashed green border
                    bbox.set_edgecolor('green')
                    bbox.set_linewidth(6)
                    bbox.set_linestyle('--')
                else:
                    # Node is not recommended as false: thick dashed red border
                    bbox.set_edgecolor('red')
                    bbox.set_linewidth(6)
                    bbox.set_linestyle('--')

    # Create legend
    legend_elements = []
    legend_elements.append(
        mpatches.Patch(facecolor='lightblue', edgecolor='blue', 
                        linewidth=4, label=f'Prefix Trace ({len(prefix_trace_features)} features)')
    )
    legend_elements.append(
        mpatches.Patch(facecolor='lightgreen', edgecolor='green', 
                        linewidth=4, linestyle='--', 
                        label=f'Recommended as True')
    )
    legend_elements.append(
        mpatches.Patch(facecolor='lightcoral', edgecolor='red', 
                        linewidth=4, linestyle='--', 
                        label='Recommended as False')
    )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20,
        frameon=True, shadow=True)
    
    # Add title with summary
    title = 'Decision Tree with Trace Prefix and Recommendations'
    if prefix_trace_features or recommended_conditions:
        title += f'\n[Prefix: {len(prefix_trace_features)} features | '
        title += f'Recommendations: {len(recommended_conditions)} conditions]'
    plot.title(title, fontsize=24, pad=20, fontweight='bold')
    
    plot.tight_layout()
    
    # Save if requested
    if save:
        plot.savefig(path, bbox_inches='tight', dpi=150)
        print(f"Tree visualization saved to: {path}")
    
    plot.show()

def plot_confusion_matrix(true_labels: list, predicted_labels: list, save: bool=False, path:str="docs/media/confusion_matrix.png"): 
    '''
        Plot the confusion matrix using matplotlib.
        Parameters:
            true_labels: The true labels of the test set.
            predicted_labels: The predicted labels from the model.
    '''
    cm_display = ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels, cmap=plot.cm.Blues)
    if save:
        cm_display.figure_.savefig(path)

def compute_all_metrics(true_labels: list, predicted_labels: list):
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

def print_recommendations_metrics(recommendations_metrics: dict):
    '''
        Print the recommendations evaluation metrics.
        Parameters:
            recommendations_metrics (dict): A dictionary containing evaluation metrics.
    '''
    print("Recommendations Evaluation Metrics:")
    for metric, value in recommendations_metrics.items():
        print(f"{metric.capitalize()}: {value*100:.2f}%")

def path_to_rule(path: Path):
    '''
        Convert a path from the decision tree to a human-readable rule.
        Each node condition (feature_name operator threshold) is combined using AND to form the rule.
            Parameters:
                path: A list of triples (feature_name, operator, threshold) representing the path.
            Returns:
                str: A human-readable rule in the form of a boolean expression.
    '''
    rule_parts = []
    for condition in path:
        if isinstance(condition, BooleanCondition):
            rule_parts.append(f"({condition.feature} == {condition.value})")
        else:
            rule_parts.append(f"({condition.feature} {condition.op} {condition.threshold})")
    return " AND ".join(rule_parts)

def print_recommendations(recommendations: dict, max_display: int = 5):
    '''
        Print the recommendations in a human-readable format.
        Parameters:
            recommendations: A dictionary where keys are prefix features (tuples) and values are sets of recommended conditions.
            max_display: Maximum number of recommendations to display.
    '''
    print(f"Displaying up to {max_display} recommendations:")
    for i, (prefix, recommendation) in enumerate(list(recommendations.items())[:max_display]):
        if len(recommendation):
            rec_str = [f"{c.feature} == {c.value}" for c in recommendation]
            print(f"Prefix Trace {i+1}: {set(prefix)} -> Recommended Activities: {rec_str}")
        else: 
            print(f"Prefix Trace {i+1}: {set(prefix)} -> No changes recommended.")
        