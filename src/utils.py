import pm4py
import logging
from pm4py.objects.log.obj import EventLog, Trace
import pm4py.objects.conversion.log.converter as log_converter
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import f1_score
from src.plotting import path_to_rule

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='w',       # replace the log file on each run
    level=logging.DEBUG,       
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        'criterion': criterion_options[best['criterion']],
        'random_state': 42
    }
    # Print the best hyperparameters
    print("Best hyperparameters:")
    print("Max Depth:", best_params['max_depth'])
    print("Max Features:", best_params['max_features'])
    print("Criterion:", best_params['criterion'])
    return best_params

def get_positive_paths(tree: DecisionTreeClassifier, feature_names: list) -> list:
    '''
        Extract all paths from root to leaves that predict the positive_class.
        To do this, simply traverse the trained decision tree using Depth-First Search (DFS).

            Parameters:
                tree: The decision tree model.
                feature_names: List of feature names used in the model.
                positive_class_value: The class value considered as positive (true).
            Returns:
                list of tuples: [(path_conditions, confidence), ...]
                where path_conditions is a list of (feature_name, boolean_value)
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

            # Recursively traverse both the left and right child nodes
            DFS_traverse_tree(tree_.children_left[node], current_path + [(feature_name, False)])
            DFS_traverse_tree(tree_.children_right[node], current_path + [(feature_name, True)])
        else:
            # Leaf node: check the predicted class
            predicted_class = bool(np.argmax(tree_.value[node][0]))
            
            # If the prediction is positive, compute confidence and store the path
            if predicted_class:
                values = tree_.value[node][0]
                total_samples = sum(values)
                positive_samples = values[1]  # assuming positive class is index 1
                confidence = positive_samples / total_samples if total_samples > 0 else 0.0
                paths.append((current_path, confidence))
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
        for feature_name, boolean_value in path[0]:
            if feature_name in prefix_trace:
                # If the condition is not satisfied, mark the path as non-compliant and break
                if prefix_trace[feature_name] != boolean_value:
                    #print(f"{feature_name}: {boolean_value} != {prefix_value} -> Discarded trace")
                    match = False
                    break

        # If the path is compliant, add it to the list
        if match:
            compliant_paths.append(path)
    return compliant_paths

def extract_recommendations(tree, feature_names, prefix_set: pd.DataFrame) -> dict:
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
    recommendation = {}

    logging.info("Extracting recommendations from the decision tree.")
    # Extract the positive paths from the decision tree
    paths = get_positive_paths(tree, feature_names)
    logging.info(f"Total positive paths extracted: {len(paths)}")
    for p in paths:
        logging.debug(f"Positive Path: {path_to_rule(p[0])} with confidence {p[1]}")
    
    # For every prefix_trace with False label
    for idx, row in prefix_set.iterrows():
        prefix_trace = row.to_dict()

        # Skip all the positive traces
        if prefix_trace.get('predicted_label') == 'true':
            logging.debug(f"Skipping positive trace: {prefix_trace.get('trace_id')}, predicted_label = {prefix_trace.get('predicted_label')}")
            continue

        logging.debug(f"Processing trace: {prefix_trace.get('trace_id')}; Full Trace: {prefix_trace}")
        # Keep for each trace only the true-valued activity
        prefix_trace_features = {
            k: v for k, v in prefix_trace.items()
            if k != 'predicted_label' and k != 'trace_id' and v != False
        }
        logging.debug(f"Prefix Trace Features: {prefix_trace_features}")
        
        # Get the compliant paths for the current prefix_trace
        compliant_paths = get_compliant_paths(paths, prefix_trace_features)
        logging.debug(f"Number of compliant paths found: {len(compliant_paths)}")
        for cp in compliant_paths:
            logging.debug(f"Compliant Path: {path_to_rule(cp[0])}, with confidence {cp[1]}")

        # If no compliant path is found, we can continue
        if not compliant_paths:
            recommendation[prefix_trace.get('trace_id')] = set()
            continue

        best_path, confidence = max(compliant_paths, key=lambda path: path[1])
        logging.info(f"Best Compliant Path: {path_to_rule(best_path)} with confidence {confidence}")

        # Extract missing conditions
        missing_conditions = {
            (feat, val)
            for (feat, val) in best_path
            if feat not in prefix_trace_features
        }

        recommendation[frozenset(prefix_trace_features)] = missing_conditions

    for k, v in recommendation.items():
        logging.debug(f"Prefix Trace: {set(k)} -> Recommended Activities: {v}")
    
    return recommendation
        
def evaluate_recommendations(test_set: pd.DataFrame, recommendations: dict) -> dict:
    """
        Evaluate the recommendations against the test set.
            Parameters:
                test_set (pd.DataFrame): The boolean encoded test set (full traces).
                recommendations (dict): A dictionary mapping prefix trace features (frozenset) 
                                    to a set of recommended activities.
            Returns:
                dict: A dictionary containing evaluation metrics
    """
    
    # Initialize counters
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # For evaluation, we need to track which traces we actually made recommendations for
    evaluated_traces = 0
    
    # Process each trace in the test set
    for idx, row in test_set.iterrows():
        trace_data = row.to_dict()
        trace_id = trace_data['trace_id']
        ground_truth = trace_data['label']
        
        # Extract the features of the full trace (excluding metadata)
        full_trace_features = {
            k: v for k, v in trace_data.items() 
            if k not in ['trace_id', 'label'] and v == True
        }
        
        # Find if we have a recommendation for this trace (by matching prefix features)
        recommendation_found = False
        recommendation_set = None
        
        for prefix_features, rec_activities in recommendations.items():
            # Check if this prefix is a subset of the full trace features
            if set(prefix_features).issubset(set(full_trace_features.keys())):
                recommendation_found = True
                recommendation_set = rec_activities
                break
        
        if not recommendation_found:
            # If no recommendation was made for this trace, skip it in evaluation
            logging.debug(f"Trace ID: {trace_id} has no recommendation. Skipping.")
            continue
        
        evaluated_traces += 1
        
        # Check if the recommendation was followed
        # A recommendation is followed if ALL recommended activities are present in the full trace
        recommendation_followed = True
        
        # Evaluate each recommended activity
        for activity, should_be_present in recommendation_set:
            if should_be_present:
                # If the activity should be present, check if it's in the full trace
                if activity not in full_trace_features:
                    recommendation_followed = False
                    break
            else:
                # If the activity should NOT be present, check if it's NOT in the full trace
                if activity in full_trace_features:
                    recommendation_followed = False
                    break

        logging.debug(f"Trace ID: {trace_id}, Ground Truth: {ground_truth}, Recommendation Followed: {recommendation_followed}")

        # Classify based on the criteria
        if recommendation_followed and ground_truth == 'true':
            true_positives += 1
        elif not recommendation_followed and ground_truth == 'false':
            true_negatives += 1
        elif recommendation_followed and ground_truth == 'false':
            false_positives += 1
        elif not recommendation_followed and ground_truth == 'true':
            false_negatives += 1
    
    # Calculate metrics
    total_predictions = true_positives + true_negatives + false_positives + false_negatives
    logging.debug(f"Total predictions: {total_predictions}")
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
    
    # Return comprehensive results
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score_value, 4),
        'accuracy': round(accuracy, 4),
    }
