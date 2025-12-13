import os
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

# Configure root logger
logging.basicConfig(
    filename='app.log',
    filemode='w',
    level=logging.WARNING,  # Set root to WARNING. Suppressing other libraries root
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Create and configure your application's logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set your module's logger to INFO

def import_log(file_path: str) -> EventLog:
    """
        Import a XES log file and convert it to an EventLog object.
        Parameters:
            file_path (str): The path to the XES log file.
        Returns:
            elem.EventLog: The imported event log object.
    """
    # Check that the file path is valid
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}. Have you extracted the dataset?")
    
    logger.info(f"Importing log from file: {file_path}")
    log = pm4py.read_xes(file_path)
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters={})
    logger.info(f"Log imported successfully with {len(log)} traces.")
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
    logger.info(f"Creating prefixes log with prefix length: {prefix_length}")
    prefixes_log = EventLog()
    
    for trace in log:
        # Create a new trace for the prefix
        prefix_trace = Trace()
        # Set the attributes of the prefix trace to match the original trace
        prefix_trace.attributes.update(trace.attributes)

        # Keep only the first 'prefix_length' events
        for event in trace[:prefix_length]:
            prefix_trace.append(event)
        # Add the trace to the new log
        prefixes_log.append(prefix_trace)
    logger.info(f"Prefixes log created successfully with {len(prefixes_log)} traces.")
    return prefixes_log

def get_activity_names(log: EventLog) -> list[str]:
    '''
        Extract unique activity names from the event log.
        Parameters:
            log (EventLog): The event log object.
        Returns:
            list[str]: A list of unique activity names.
    '''
    logger.info("Extracting activity names from log.")
    activity_names=[]
    # Iterate through each trace and event to collect activity names
    for trace in log:
        for event in trace:
            activity_names.append(event['concept:name'])
    # Remove duplicates while preserving order
    return sorted(set(activity_names), key=lambda x:activity_names.index(x))

def compute_columns(activity_names:list) -> list[str]:
    '''
        Returns a list of column names for boolean encoding. The columns include:
            - 'trace_id' for trace identification;
            - One column for each activity name;
            - 'label' for the ground truth label.
        Parameters:
            activity_names (list): List of unique activity names.
        Returns:
            list: List of column names including 'trace_id', activity names, and 'label'.
    '''
    # Start with 'trace_id' column
    columns = ['trace_id']
    # Add a column representing the length of the prefix
    columns += ['prefix_length']
    # Add the found activity names as columns
    columns += activity_names
    # Add 'label' column for ground truth
    columns.append('label')
    return columns

def boolean_encode(log: EventLog, activity_names:list):
    '''
        Boolean encode the event log into a DataFrame. For each trace, create a row with:
            - 'trace_id' for trace identification;
            - One boolean column for each activity name indicating presence (True) or absence (False);
            - 'label' for the ground truth label.
        Parameters:
            log (EventLog): The event log object.
            activity_names (list): List of activity names.
        Returns:
            pd.DataFrame: Boolean encoded DataFrame of the event log.
    '''
    logger.info("Creating boolean encoding")
    encoded_log = []
    # Build column names
    columns = compute_columns(activity_names)
    
    # For each trace in the log
    for trace in log:
        # Initialize the encoded row with trace_id
        encoded_row = [trace.attributes["concept:name"]]
        # Append the prefix length
        encoded_row.append(len(trace))
        logger.debug(f"Encoding trace ID: {trace.attributes['concept:name']} with prefix length: {len(trace)}")
        # Initialize boolean indicators for each activity as False
        bool_events = [False]*len(activity_names)   
        for event in trace:
            # Get the activity name of the event
            event_name = event["concept:name"]
            if event_name in activity_names:
                # Get the index of the activity name
                activity_name_index = activity_names.index(event["concept:name"])
                # Mark the activity as present (True)
                bool_events[activity_name_index]=True
        # Append boolean indicators and label to the encoded row
        encoded_row += bool_events
        
        # Append the ground truth label to the row
        encoded_row.append(trace.attributes["label"])
        
        # Add the encoded row to the log
        encoded_log.append(encoded_row)
    
    logger.info("Boolean encoding created successfully.")
    return pd.DataFrame(columns=columns, data=encoded_log)

def hyperparameter_optimization(encoded_data:pd.DataFrame, max_evals:int=100, space:dict=None) -> dict:
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
    if space is None:
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
        Once a positive leaf is reached, compute the confidence of that path and store the path.
            Parameters:
                tree: The decision tree model.
                feature_names: List of feature names used in the model.
                positive_class_value: The class value considered as positive (true).
            Returns:
                list of tuples: [(path_conditions, confidence), ...]
                where path_conditions is a list of tuples (feature_name, boolean_value)
        
        TODO: refactor this, adding the support for prefix_length conditions in the path. We should restore the old
        path definition (feature, operator, value)
    '''
    logger.info("Extracting positive paths from the decision tree.")
    tree_ = tree.tree_
    paths = []

    '''
        Depth-First Search (DFS) traversal of the decision tree to find positive paths.
        This function is called recursively to explore all paths from the root to the leaves.
            Parameters:
                node: The current node in the decision tree.
                current_path: The path taken to reach the current node.
        If we reach a positive leaf, we store the current path in the paths list, with its confidence   .
    '''
    def DFS_traverse_tree(node, current_path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            # Base case: leaf node
            # Check the predicted class
            predicted_class = bool(np.argmax(tree_.value[node][0]))
            
            # If the prediction is positive, compute confidence and store the path
            if predicted_class:
                values = tree_.value[node][0]
                total_samples = sum(values)
                positive_samples = values[1]  # assuming positive class is index 1
                confidence = positive_samples / total_samples if total_samples > 0 else 0.0 
                # Store the path and its confidence
                paths.append((current_path, confidence))
                logger.debug(f"Found positive path: {path_to_rule(current_path)} with confidence {confidence}")
            return
        else:
            # Recursive case: internal node
            # Get the feature name for the current node
            feature_name = feature_names[tree_.feature[node]]

            # Recursively traverse both the left and right child nodes
            DFS_traverse_tree(tree_.children_left[node], current_path + [(feature_name, False)]) # Taking the left path: the feature is not present
            DFS_traverse_tree(tree_.children_right[node], current_path + [(feature_name, True)]) # Taking the right path: the feature is present
            
    # Start the recursive DFS traversal, with an empty path
    DFS_traverse_tree(0, [])
    logger.info(f"Extracted {len(paths)} positive paths from the decision tree.")
    return paths
    
def get_compliant_paths(paths: list, prefix_trace: dict) -> list:
    '''
        Extract the paths that are compliant with the given prefix_trace. A path is compliant
        if no condition in the path contradicts the prefix_trace (no True features in the prefix that are False in the path).
            Parameters:
                paths: A list of paths to filter.
                prefix_trace: A dictionary representing only the activity done in the prefix trace.
            Returns:
                list: filter list of paths, containing only the compliant ones.    
        TODO: add another condition: if the path contains a condition on prefix_length, check that the length of the prefix_trace is smaller
    '''
    logger.info("Extracting compliant paths for the given prefix trace.")
    compliant_paths = []
    
    # For each positive path
    for path in paths:
        match = True
        
        # For each condition in the path
        for feature_name, boolean_value in path[0]:
            # If that feature was done in the prefix trace
            if feature_name in prefix_trace:
                # If the condition is not satisfied, mark the path as non-compliant and break
                if prefix_trace[feature_name] != boolean_value:
                    logger.debug(f"{feature_name}: {boolean_value} != {prefix_trace[feature_name]} (prefix) -> Discarded trace")
                    match = False
                    break

        # If the path is compliant, add it to the list
        if match:
            compliant_paths.append(path)
            logger.debug(f"Compliant path found: {path_to_rule(path[0])} with confidence {path[1]}")

    logger.info(f"Found {len(compliant_paths)} compliant paths for the given prefix trace.")
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
                if the prefix is already positive, the recommendation is an empty set.
                Otherwise, where possible, it contains the missing activities to reach a positive outcome.
    '''
    logger.info("Extracting recommendations for the given prefix set.")
    recommendation = {}

    # Extract the positive paths from the decision tree
    paths = get_positive_paths(tree, feature_names)
        
    # For every prefix_trace with False label
    for idx, row in prefix_set.iterrows():
        prefix_trace = row.to_dict()

        # If the prefix trace was predicted as positive, we can add an empty recommendation
        if prefix_trace.get('predicted_label') == 'true':
            true_prefix = frozenset({k: v for k, v in prefix_trace.items() if k != 'predicted_label' and k != 'trace_id' and v})
            recommendation[true_prefix] = set()
            continue

        logger.debug(f"Processing trace: {prefix_trace.get('trace_id')}; Full Trace: {prefix_trace}")
        # Keep for each trace only the true-valued activity
        prefix_trace_features = {
            k: v for k, v in prefix_trace.items()
            if k != 'predicted_label' and k != 'trace_id' and v != False
        }
        logger.debug(f"Prefix Trace Features: {prefix_trace_features}")
        
        # Get the compliant paths for the current prefix_trace
        compliant_paths = get_compliant_paths(paths, prefix_trace_features)

        # If no compliant path is found, set empty recommendation
        if not compliant_paths:
            recommendation[prefix_trace.get('trace_id')] = set()
            continue

        # Pick the path with highest confidence, break ties by shortest length
        best_path, confidence = max(
            compliant_paths,
            key=lambda x: (x[1], -len(x[0]))  # x[1] = confidence, x[0] = path list
        )
        logger.info(f"Best Compliant Path: {path_to_rule(best_path)} with confidence {confidence}")

        # Extract missing conditions
        missing_conditions = {
            (feat, val)
            for (feat, val) in best_path
            if feat not in prefix_trace_features
        }

        recommendation[frozenset(prefix_trace_features)] = missing_conditions

    for k, v in recommendation.items():
        logger.debug(f"Prefix Trace: {set(k)} -> Recommended Activities: {v}")
    
    logger.info("Recommendations extraction completed.")
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
    logger.info("Evaluating recommendations")
    
    
    # Initialize counters
    t_p = t_n = f_p = f_n = 0
    
    # Process each trace in the test set
    for _, row in test_set.iterrows():
        trace_id = row['trace_id']
        ground_truth = row['label']
        
        # Extract the features of the full trace
        full_trace_features = {
            k: v for k, v in row.items() 
            if k not in ['trace_id', 'label', 'prefix_length'] and v == True
        }
        
        # Find matching recommendation for this prefix trace
        recommendation = None

        for prefix_features, rec in recommendations.items():
            # Check if this prefix is a subset of the full trace features
            # A prefix 'P' matches if all activities in 'P' are also in the 'full_trace_features'.
            if set(prefix_features).issubset(set(full_trace_features)):
                recommendation = rec
                break
        
        if recommendation is None:
            # If no recommendation was made for this trace, skip it in evaluation
            logger.debug(f"Trace ID: {trace_id} has no recommendation. Skipping.")
            continue
        
        """
        Check if the recommendation was followed. 
         - A recommendation is followed if all recommended activities are present (True)
         - and all recommended activities that should be absent (False) are indeed absent in the full trace.
        """
        recommendation_followed = True
        
        # Evaluate each recommended activity
        for activity, should_be_present in recommendation:
            if should_be_present and activity not in full_trace_features:
                recommendation_followed = False
                break
            if not should_be_present and activity in full_trace_features:
                recommendation_followed = False
                break
        
        logger.debug(f"Trace {trace_id}: truth: {ground_truth}, Recommendation Followed: {recommendation_followed}")

        # Classify based on the report's criteria (Section 2.6)
        if recommendation_followed and ground_truth == 'true':
            # True Positives: The recommended activity was followed in the actual trace, and the ground truth outcome is positive. 
            t_p += 1
        elif not recommendation_followed and ground_truth == 'false':
            # True Negatives: The recommended activity was not followed in the actual trace, and the ground truth outcome is negative. 
            t_n += 1
        elif not recommendation_followed and ground_truth == 'true':
            # False Positives: The recommended activity was not followed in the actual trace, but the ground truth outcome is positive. 
            f_p += 1
        elif recommendation_followed and ground_truth == 'false':
            # False Negatives: The recommended activity was followed in the actual trace, but the ground truth outcome is negative. 
            f_n += 1

    # Calculate metrics
    total_predictions = t_p + t_n + f_p + f_n
    
    if total_predictions == 0:
         return {
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'accuracy': 0.0,
        }

    precision = t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0
    recall = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (t_p + t_n) / total_predictions

    # Return comprehensive results
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score_value, 4),
        'accuracy': round(accuracy, 4),
    }
