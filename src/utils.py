import os
import pm4py
import logging
from pm4py.objects.log.obj import EventLog, Trace
import pm4py.objects.conversion.log.converter as log_converter
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import f1_score, accuracy_score
from src.plotting import path_to_rule
from src.types import BooleanCondition, ThresholdCondition, Condition, Path

# Configure root logger
logging.basicConfig(
    filename='app.log',
    filemode='w',
    level=logging.WARNING,  # Set root to WARNING. Suppressing other libraries root
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create and configure your application's logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set your module's logger to DEBUG

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
    if prefix_length < 1:
        logger.error("Prefix length must be at least 1.")
        raise ValueError("Prefix length must be at least 1.")
    
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
    activity_names = sorted(set(activity_names), key=lambda x:activity_names.index(x))
    logger.debug(f"Extracted activity names: {activity_names}")
    return activity_names

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
    columns = compute_columns(activity_names)
    
    # For each trace in the log
    for trace in log:
        # Initialize the encoded row with trace_id and prefix length
        encoded_row = [trace.attributes["concept:name"]]
        encoded_row.append(len(trace))       
        # Initialize boolean indicators for each activity as False
        bool_events = [False] * len(activity_names)
        for event in trace:
            # Get the activity name of the event
            event_name = event["concept:name"]
            if event_name in activity_names:
                # Get the index of the activity name
                activity_name_index = activity_names.index(event["concept:name"])
                # Mark the activity as present (True)
                bool_events[activity_name_index] = True
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
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(0))
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

def exclude_keys_from_trace(trace: dict, keys_to_exclude: set=['trace_id', 'label', 'prefix_length'], remove_false_activities: bool=True) -> dict:
    '''
        Exclude specified keys from a trace dictionary, returning only keys with True values.
            Parameters:
                trace (dict): The original trace dictionary.
                keys_to_exclude (set): Set of keys to exclude from the trace.
            Returns:
                dict: The trace dictionary with specified keys excluded.
    '''
    return {k: v for k, v in trace.items() if k not in keys_to_exclude and (v == True or not remove_false_activities)}

def is_leaf(tree_, node_id: int) -> bool:
    '''
        Check if a node in the decision tree is a leaf node.
            Parameters:
                node: The decision tree object.
                node_id: The ID of the node to check.
            Returns:
                bool: True if the node is a leaf, False otherwise.
    '''
    return tree_.feature[node_id] == _tree.TREE_UNDEFINED

def compute_confidence(tree_: DecisionTreeClassifier, node_id: int) -> float:
    '''
        Compute the confidence of a leaf node in the decision tree.
            Parameters:
                tree: The decision tree model.
                node_id: The ID of the leaf node.
            Returns:
                float: The confidence of the leaf node.
    '''
    values = tree_.value[node_id][0]
    return values[1]
    
def get_positive_paths(tree: DecisionTreeClassifier, feature_names: list) -> list[tuple[Path, float]]:
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
                Where path_conditions is a list of Condition objects representing the path.
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
        if is_leaf(tree_, node):
            # Base case: leaf node
            predicted_class = bool(np.argmax(tree_.value[node][0]))
    
            # If the prediction is positive, compute confidence and store the path
            if predicted_class:
                confidence = compute_confidence(tree_, node)
                logger.debug(f"Found positive path: {path_to_rule(current_path)} with confidence {confidence}")
                paths.append((current_path, confidence))
            return
        else:
            # Recursive case: internal node
            # Get the feature name for the current node
            feature_name = feature_names[tree_.feature[node]]

            # Check if the feature is 'prefix_length' to handle it differently
            if feature_name == 'prefix_length':
                threshold = tree_.threshold[node]
                # Left child: value <= threshold
                DFS_traverse_tree(tree_.children_left[node], current_path + [ThresholdCondition(feature=feature_name, op='<=', threshold=threshold)])
                # Right child: value > threshold
                DFS_traverse_tree(tree_.children_right[node], current_path + [ThresholdCondition(feature=feature_name, op='>', threshold=threshold)])
                return
            # Left child: feature is False
            DFS_traverse_tree(tree_.children_left[node], current_path + [BooleanCondition(feature=feature_name, value=False)])
            # Right child: feature is True
            DFS_traverse_tree(tree_.children_right[node], current_path + [BooleanCondition(feature=feature_name, value=True)])

    # Start the recursive DFS traversal, with an empty path
    DFS_traverse_tree(0, [])
    logger.info(f"Extracted {len(paths)} positive paths from the decision tree.")
    return paths

def contradicts(condition: Condition, prefix_trace: dict) -> bool:
    '''
        Check if a given condition is contradicted by the prefix_trace.
            Parameters:
                condition: A Condition object (BooleanCondition or ThresholdCondition).
                prefix_trace: A dictionary representing the current prefix trace.
            Returns:
                bool: True if the condition is contradicted, False otherwise.
    '''
    # If the condition is a BooleanCondition, check if the feature value matches
    if isinstance(condition, BooleanCondition):
        # Missing feature in prefix_trace means no contradiction
        if condition.feature not in prefix_trace:
            return False
        return prefix_trace.get(condition.feature) != condition.value
    # If the condition is a ThresholdCondition, check the threshold condition
    else:
        value = prefix_trace.get(condition.feature)
        if condition.op == '<=':
            return value > condition.threshold
        return value <= condition.threshold

def get_compliant_paths(paths: list[tuple[Path, float]], prefix_trace: dict) -> list[tuple[Path, float]]:
    '''
        Extract the paths that are compliant with the given prefix_trace. A path is compliant
        if no condition in the path contradicts the prefix_trace (no True features in the prefix that are False in the path).
            Parameters:
                paths: A list of paths to filter.
                prefix_trace: A dictionary representing only the activity done in the prefix trace.
            Returns:
                list: filter list of paths, containing only the compliant ones.    
    '''
    logger.info("Extracting compliant paths.")
    compliant = []
    
    # For each positive path
    for path, confidence in paths:
        # Check if no condition in the path is contradicted by the prefix_trace
        if not any(contradicts(c, prefix_trace) for c in path):
            compliant.append((path, confidence))
            logger.debug(f"Compliant path: {path_to_rule(path)} ({confidence:.3f})")

    logger.info(f"Found {len(compliant)} compliant paths.")
    return compliant

def get_missing_conditions(best_path: Path, prefix_trace: dict) -> list[BooleanCondition]:
    '''
        Extract the missing conditions from the best_path that are not satisfied by the prefix_trace.
            Parameters:
                best_path: The best path (list of Condition objects).
                prefix_trace: A dictionary representing the current prefix trace.
            Returns:
                list: A list of BooleanCondition objects representing the missing conditions.
    '''
    recommendations = []

    for condition in best_path:
        if isinstance(condition, BooleanCondition):
            if prefix_trace.get(condition.feature) != condition.value:
                recommendations.append(condition)
    return recommendations

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
        
    # For every prefix_trace
    for _, row in prefix_set.iterrows():
        prefix_trace = row.to_dict()
        trace_id = prefix_trace.get('trace_id')
        ground_truth = prefix_trace.get('predicted_label')
        logger.debug(f"Processing trace: {trace_id}; predicted as: {ground_truth}")

        # Skip already positive traces
        if ground_truth == 'true':
            logger.debug(f"Trace {trace_id} is already positive; no recommendation needed.")
            recommendation[frozenset(exclude_keys_from_trace(prefix_trace).keys())] = None
            continue
        
        # Extract true valued activity features
        true_features = exclude_keys_from_trace(prefix_trace)
        prefix_trace_key = frozenset(true_features.keys())
        
        # Current prefix conditions (true features + prefix_length)
        current_prefix_conditions = {
            **true_features,
            "prefix_length": row["prefix_length"]
        }
        logger.debug(f"Current Prefix Conditions: {current_prefix_conditions}")
        
        # Get the compliant paths for the current prefix_trace
        compliant_paths = get_compliant_paths(paths, current_prefix_conditions)

        # If no compliant path is found, set empty recommendation
        if not compliant_paths:
            logger.debug(f"No compliant paths found for trace {trace_id}; no recommendation possible.")
            recommendation[prefix_trace_key] = set()
            continue

        # Pick the path with highest confidence, break ties by shortest length
        best_path, confidence = max(
            compliant_paths,
            key=lambda item: (item[1], -len(item[0]))  # item[1] = confidence, item[0] = path list
        )
        logger.info(f"Best Compliant Path: {path_to_rule(best_path)} with confidence {confidence}")
        
        # Extract missing conditions as recommendations
        recommendation[prefix_trace_key] = get_missing_conditions(best_path, current_prefix_conditions)

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
    logger.debug(f"Total recommendations to evaluate: {len(recommendations)}")
    
    # Pre-compute feature sets for all traces in the test set
    test_traces = []
    for _, trace in test_set.iterrows():
        trace_features = frozenset(exclude_keys_from_trace(trace.to_dict()))
        test_traces.append((trace_features, trace))

    # Initialize counters
    t_p = t_n = f_p = f_n = 0

    # For each trace in the test set and its corresponding recommendation
    for (trace_features, full_trace), (_, recommendation) in zip(test_traces, recommendations.items()):
        trace_id = full_trace['trace_id']
        ground_truth = full_trace['label']
                
        # If the recommendation are None, it means the trace was already positive
        if recommendation is None:
            logger.debug(f"Trace {trace_id} was already positive; skipping recommendation evaluation.")
            continue
        
        # If no recommendation was possible, skip the trace
        if recommendation == set():
            logger.debug(f"Trace {trace_id} has negative outcome, but no recommendation was possible.")
            continue
            
        # Check if the recommendation was followed in the full trace
        recommendation_followed = True
        for boolean_condition in recommendation:
            activity = boolean_condition.feature

            # Checking if the activity should be present in the test trace
            should_be_present = boolean_condition.value
            logger.debug(f"Trace {trace_id}: Checking recommendation for activity '{activity}' to be {'present' if should_be_present else 'absent'}")
            
            # Check if the activity is present in the test trace
            is_present = activity in trace_features
            logger.debug(f"Trace {trace_id}: Activity '{activity}' is {'present' if is_present else 'absent'} in the full trace")
            
            # If any condition is not met, the recommendation is not followed
            if (should_be_present and not is_present or 
              not should_be_present and is_present):
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
            'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0,
            'f1_score': 0.0, 'accuracy': 0.0,
        }

    precision = t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0
    recall = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (t_p + t_n) / total_predictions

    # Return comprehensive results
    return {
        'tp': t_p,
        'tn': t_n,
        'fp': f_p,
        'fn': f_n,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score_value, 4),
        'accuracy': round(accuracy, 4),
    }

def compute_prefix_length_statistics(training_log: EventLog, test_log: EventLog, prefix_length: int, activity_names: list, hyperparameter_space: dict, max_evals: int=300) -> dict:
    '''
            Parameters:
                - training_log (EventLog): The training event log object.
                - test_log (EventLog): The test event log object.
                - prefix_length (int): The prefix length to compute statistics for.
                - activity_names (list): List of activity names.
                - hyperparameter_space (dict): The hyperparameter search space for optimization.
                - max_evals (int): The maximum number of evaluations for hyperparameter optimization.
            Returns:
                dict: A dictionary containing statistics about the given prefix length. It is 
                composed by:
                    - 'tree_accuracy': Accuracy of the Decision Tree Classifier.
                    - 'tree_f1': F1-score of the Decision Tree Classifier.
                    - 'recommendation_accuracy': Accuracy of the recommendation system.
                    - 'recommendation_f1': F1-score of the recommendation system.
    '''
    logger.info(f"Computing statistics for prefix length {prefix_length}.")
    result = {}
    # Prepare training data
    pruned_log = create_prefixes_log(training_log, prefix_length=prefix_length)
    encoded_log = boolean_encode(pruned_log, activity_names)
    features = ['prefix_length'] + activity_names

    # Hyperparameter optimization
    params = hyperparameter_optimization(encoded_log, max_evals=max_evals, space=hyperparameter_space)
    clf = DecisionTreeClassifier(
        max_depth=params['max_depth'], 
        max_features=params['max_features'], 
        criterion=params['criterion'], 
        random_state=params['random_state']
    )
    clf.fit(encoded_log.drop(['trace_id', 'label'], axis=1), encoded_log['label'])

    # Prepare test data
    test_log_prefix = create_prefixes_log(test_log, prefix_length=prefix_length)
    test_encoded_log = boolean_encode(test_log_prefix, activity_names)
    predictions = clf.predict(test_encoded_log.drop(['trace_id', 'label'], axis=1))
    
    # Compute evaluation metrics
    true_labels = test_encoded_log['label'].values
    pred_accuracy = accuracy_score(true_labels, predictions)
    pred_f1 = f1_score(true_labels, predictions, pos_label='true')
    # Store results
    result['tree_accuracy'] = pred_accuracy
    result['tree_f1'] = pred_f1
    
    # Compute the recommendation evaluation metrics
    test_encoded_log_with_predictions = test_encoded_log.copy().drop('label', axis=1)
    test_encoded_log_with_predictions['predicted_label'] = predictions
    full_trace_test_encoded_log = boolean_encode(test_log, activity_names)
    recommendations = extract_recommendations(clf, features, test_encoded_log_with_predictions)
    evaluation_metrics = evaluate_recommendations(full_trace_test_encoded_log, recommendations)
    result['recommendation_accuracy'] = evaluation_metrics['accuracy']
    result['recommendation_f1'] = evaluation_metrics['f1_score']
    logger.info(f"Statistics for prefix length {prefix_length}: {result}")
    return result