import pm4py
from pm4py.objects.log.obj import EventLog, Trace
import pm4py.objects.conversion.log.converter as log_converter
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import f1_score

def import_log(file_path: str) -> EventLog:
    """
        Import a XES log file and convert it to an EventLog object.
        Parameters:
            file_path (str): The path to the XES log file.
        Returns:
            elem.EventLog: The imported event log object.
    """
    log = pm4py.read_xes(file_path)
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters={})
    return log

def create_prefixes_log(log: EventLog, prefix_length: int) -> EventLog:
    """
        Creates a new log object containing for each trace 
        only the first 'prefix_length' events.
        Parameters:
            log (EventLog): The original event log object.
            prefix_length (int): The length of the prefixes to extract.
        Returns:
            EventLog: The new event log object with prefixes.
    """
    prefixes_log = EventLog()
    
    for trace in log:
        # Create a new trace for the prefix
        prefix_trace = Trace()
        # Set the attributes of the prefix trace to match the original trace
        prefix_trace.attributes.update(trace.attributes)

        # For each of the first 'prefix_length' events in the trace
        for event in trace[:prefix_length]:
            prefix_trace.append(event)
        # Add the trace to the new log
        prefixes_log.append(prefix_trace)
    return prefixes_log

def get_activity_names(log: EventLog) -> list[str]:
    '''
        Extract unique activity names from the event log.
        Parameters:
            log (EventLog): The event log object.
        Returns:
            list[str]: A list of unique activity names.
    '''
    activity_names=[]
    # Iterate through each trace and event to collect activity names
    for trace in log:
        for event in trace:
            activity_names.append(event['concept:name'])
    # Remove duplicates while preserving order
    return sorted(set(activity_names), key=lambda x:activity_names.index(x))

def compute_columns(activity_names:list) -> list[str]:
    '''
        Compute the column names for the dataset based on activity names.
        Parameters:
            activity_names (list): List of activity names.
        Returns:
            list: List of column names including 'trace_id', activity names, and 'label'.
    '''
    # Start with 'trace_id' column
    columns = ['trace_id']
    # Add the found activity names as columns
    columns += activity_names
    # Add 'label' column for ground truth
    columns.append('label')
    return columns

def boolean_encode(log: EventLog, activity_names:list):
    '''
        Boolean encode the event log into a DataFrame.
        Parameters:
            log (EventLog): The event log object.
            activity_names (list): List of activity names.
        Returns:
            pd.DataFrame: Boolean encoded DataFrame of the event log.
    '''
    encoded_log = []
    # Build column names
    columns = compute_columns(activity_names)
    for trace in log:
        # Initialize the encoded row with trace_id
        encoded_row = [trace.attributes["concept:name"]]
        # Initialize boolean indicators for each activity
        bool_events = [False]*len(activity_names)   
        for event in trace:
            event_name = event["concept:name"]
            # If the event name is present in activity names, that means this activity occurred in the trace
            if event_name in activity_names:
                # Get the index of the activity name in order to update the corresponding boolean indicator
                activity_name_index = activity_names.index(event["concept:name"])
            bool_events[activity_name_index]=True
        # Append boolean indicators and label to the encoded row
        encoded_row += bool_events
        encoded_row.append(trace.attributes["label"])
        # Append the encoded row to the encoded log
        encoded_log.append(encoded_row)
    return pd.DataFrame(columns=columns, data=encoded_log)

def hyperparameter_optimization(encoded_data:pd.DataFrame, max_evals:int=100) -> dict:
    '''
        Hyperparameter optimization for Decision Tree Classifier using Hyperopt.
        Parameters:
            encoded_data (pd.DataFrame): The boolean encoded event log DataFrame.
        Returns:
            dict: The best hyperparameters found.
    '''
    def fmeasure_model(params):
        '''
            Train a Decision Tree Classifier with given parameters and compute F1-measure.
            Parameters:
                params (dict): Hyperparameters for the Decision Tree Classifier.
                max_evals (int): Maximum number of evaluations for Hyperopt.
            Returns:
                dict: A dictionary with negative F1-measure as loss and status.
        '''
        X_ = encoded_data.drop(['trace_id', 'label'], axis=1)
        y = encoded_data['label']
        clf = DecisionTreeClassifier(**params)
        clf.fit(X_, y)
        # Predict on the training data to get predictions for F1-measure calculation
        y_pred = clf.predict(X_)
        fmeasure = f1_score(y, y_pred, average='weighted')
        return {'loss': -fmeasure, 'status': STATUS_OK}

    def f(params):
        '''
            Wrapper function to compute F1-measure for Hyperopt.
            Parameters:
                params (dict): Hyperparameters for the Decision Tree Classifier.
            Returns:
                float: The F1-measure of the model.
        '''
        return fmeasure_model(params)

    # Define the hyperparameter search space
    space = {
        'max_depth': hp.choice('max_depth', range(1, 400)),
        'max_features': hp.choice('max_features', range(1, 448)),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'random_state': 42
    }

    # Run hyperparameter optimization
    trials = Trials()
    # The best hyperparameters found
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    # Map the indices back to actual hyperparameter values
    criterion_options = ['gini', 'entropy']
    best_params = {
        'max_depth': best['max_depth'] + 1,  # +1 because range starts from 1
        'max_features': best['max_features'] + 1,  # +1 because range starts from 1
        'criterion': criterion_options[best['criterion']]
    }
    # Print the best hyperparameters
    print("Best hyperparameters:")
    print("Max Depth:", best_params['max_depth'])
    print("Max Features:", best_params['max_features'])
    print("Criterion:", best_params['criterion'])
    return best_params

def get_positive_paths(tree: DecisionTreeClassifier, feature_names: list, positive_class_value: str, class_values: list) -> list:
    '''
        Extract all paths from root to leaves that predict the positive_class.
        To do this, simply traverse the trained decision tree using Depth-First Search (DFS).

            Parameters:
                tree: The decision tree model.
                feature_names: List of feature names used in the model.
                positive_class_value: The class value considered as positive (true).
            Returns:
                list: A list of positive paths, where each path is represented as a list
                of tuples (feature_name, 'true'/'false'). 
    '''
    tree_ = tree.tree_
    paths = []

    '''
        Depth-First Search (DFS) traversal of the decision tree to find positive paths.
        This function is called recursively to explore all paths from the root to the leaves.
            Parameters:
                node: The current node in the decision tree.
                current_path: The path taken to reach the current node.
        If we reach a positive leaf, we store the current path in the paths list.
    '''
    def DFS_traverse_tree(node, current_path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Get the feature name for the current node
            feature_name = feature_names[tree_.feature[node]]
            # Get the threshold for the current node
            threshold = tree_.threshold[node]
            # Recursively traverse both the left and right child nodes
            DFS_traverse_tree(tree_.children_left[node], current_path + [(feature_name, class_values[0])])
            DFS_traverse_tree(tree_.children_right[node], current_path + [(feature_name, class_values[1])])
        else:
            # We are in a leaf node, check the predicted class
            predicted_class= class_values[np.argmax(tree_.value[node][0])]
            # If the prediction is positive, store the path
            if predicted_class == positive_class_value:
                paths.append(current_path)
            return
    # Start the recursive DFS traversal, with an empty path
    DFS_traverse_tree(0, [])
    return paths

def get_compliant_paths(paths: list, prefix_trace: dict) -> list:
    '''
        Extract the paths that are compliant with the given prefix_trace.
            Parameters:
                paths: A list of paths to filter.
                prefix_trace: A dictionary representing the prefix trace.
            Returns:
                list: A list of compliant paths.
    '''
    compliant_paths = []
    # For each positive path
    for path in paths:
        match = True
        
        # For each condition in the path
        for feature_name, boolean_value in path:
            # Check if the feature exists in the prefix_trace
            if feature_name not in prefix_trace:
                match = False
                break

            # Get the value of the current feature from the prefix_trace
            prefix_value = prefix_trace[feature_name]
            # If the condition is not satisfied, mark the path as non-compliant and break
            if boolean_value != prefix_value:
                match = False
                break
        # If the path is compliant, add it to the list
        if match:
            compliant_paths.append(path)
    return compliant_paths

def get_best_compliant_path(compliant_paths: list, tree: DecisionTreeClassifier, feature_names: list, class_values: list) -> tuple:
    '''
        Find the compliant path with the highest confidence.
            Parameters:
                compliant_paths: A list of compliant paths.
                tree: The decision tree model.
                feature_names: List of feature names used in the model.
                class_values: List of possible class values [negative_value, positive_value].
            Returns:
                tuple: (best_path, confidence) - The compliant path with the highest confidence and its confidence score.
    '''
    if not compliant_paths:
        return None, 0.0
    
    best_path = None
    highest_confidence = -1.0

    for path in compliant_paths:
        node = 0
        valid_traversal = True
        
        # Traverse the tree to find the leaf node corresponding to the path
        for feature_name, boolean_value in path:
            # Check if we've reached a leaf node prematurely
            if tree.tree_.feature[node] == _tree.TREE_UNDEFINED:
                valid_traversal = False
                break
            
            # Get the feature index
            feature_index = feature_names.index(feature_name)
            
            # Check if current node's feature matches
            if tree.tree_.feature[node] == feature_index:
                # Follow the appropriate branch based on boolean_value
                if boolean_value == class_values[0]:  # Left child (e.g., 'false')
                    node = tree.tree_.children_left[node]
                elif boolean_value == class_values[1]:  # Right child (e.g., 'true')
                    node = tree.tree_.children_right[node]
                else:
                    # Invalid boolean value
                    valid_traversal = False
                    break
            else:
                # Feature mismatch - this shouldn't happen for valid paths
                valid_traversal = False
                break
        
        if not valid_traversal:
            continue
        
        # Get the confidence of the prediction at the leaf node
        value = tree.tree_.value[node]
        total_samples = sum(value[0])
        positive_samples = value[0][1]  # Assuming positive class is indexed at 1
        confidence = positive_samples / total_samples if total_samples > 0 else 0

        # Update the best path if this one has higher confidence
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_path = path

    return best_path, highest_confidence

def path_to_rule(path):
    '''
        Convert a path from the decision tree to a human-readable rule.
        Each node condition (feature_name operator threshold) is combined using AND to form the rule.
            Parameters:
                path: A list of triples (feature_name, operator, threshold) representing the path.
            Returns:
                str: A human-readable rule in the form of a boolean expression.
    '''
    rule = []
    for feature_name, boolean_value in path:
        rule.append(f"{feature_name} == {boolean_value}")
    return " AND ".join(rule)

def extract_recommendations(tree, feature_names, class_values:list, prefix_set: pd.DataFrame) -> dict:
    '''
        Extract recommendations from a decision tree for given prefixes.
            Parameters:
                tree: The decision tree model.
                feature_names: List of feature names used in the model.
                class_values: Possible class values for the target variable (true, false).
                prefix_set: Set of prefixes to extract recommendations for.
            Returns:
                dict: A dictionary mapping prefixes to their recommendations.
    '''
    POSITIVE_CLASS_VALUE = class_values[1] 
    recommendation = {}

    # Extract the positive paths from the decision tree
    paths = get_positive_paths(tree, feature_names, positive_class_value=POSITIVE_CLASS_VALUE, class_values=class_values)

    # For every prefix_trace with False label
    for idx, row in prefix_set.iterrows():
        prefix_trace = row.to_dict()
        # Only process negative cases
        if prefix_trace.get('label') == POSITIVE_CLASS_VALUE:
            continue
        
        # Filter only the actual activities done in the trace
        prefix_trace_features = {k: v for k, v in prefix_trace.items() if k != 'label' and k != 'trace_id'}
        print(prefix_trace_features)

        # Get the compliant paths for the current prefix_trace
        compliant_paths = get_compliant_paths(paths, prefix_trace_features)

        # If there is at least one compliant path
        if compliant_paths:
            best_path, _ = get_best_compliant_path(compliant_paths, tree, feature_names, class_values)
        
        # Find the path with the highest confidence
    return 0
        
def evaluate_recommendations(test_set: EventLog, recommendations:list) -> dict:
    '''
        Evaluate the recommendations against the test set.
            Parameters:
                test_set(EventLog): The test dataset.
                recommendations(list): The recommendations to evaluate.
            Returns:
                dict: A dictionary containing evaluation metrics.
    '''
    # Initialize confusion matrix components
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for prefix_trace in test_set:
        trace_id = prefix_trace.attributes["concept:name"]
        ground_truth = prefix_trace.attributes["label"]
        # Skip if there are no recommendations for this trace
        if trace_id not in recommendations:
            continue
        # Get the recommended outcome for this trace
        recommended_outcome = recommendations[trace_id]
        # Get all the activities occured in the trace
        trace_activities = set([event['concept:name'] for event in prefix_trace])
        # Check if all recommended activities are followed in the trace
        recommendation_followed = all(activity in trace_activities for activity in recommended_outcome['activities'])
        # Calculating the confusion matrix components
        if recommendation_followed and ground_truth == 'positive':
            true_positive += 1
        elif recommendation_followed and ground_truth == 'negative':
            false_positive += 1
        elif not recommendation_followed and ground_truth == 'negative':
            true_negative += 1
        elif not recommendation_followed and ground_truth == 'positive':
            false_negative += 1 

        # Calculate evaluation metrics
        total = true_positive + true_negative + false_positive + false_negative
        # Precision 
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        # Recall
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        # Accuracy
        accuracy = (true_positive + true_negative) / total if total > 0 else 0
        # F1-Score
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # Group metrics into a dictionary
        metrics = {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1_score
        }
    return metrics