import pm4py
import pm4py.objects.log.obj as elem
import pm4py.objects.conversion.log.converter as log_converter
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

def import_log(file_path: str) -> elem.EventLog:
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

def get_activity_names(log: elem.EventLog) -> list[str]:
    '''
        Extract unique activity names from the event log.
        Parameters:
            log (elem.EventLog): The event log object.
        Returns:
            list[str]: A list of unique activity names.
    '''
    activity_names=[]
    # Iterate through each case and event to collect activity names
    for case in log:
        for event in case:
            activity_names.append(event['concept:name'])
    # Remove duplicates while preserving order
    return sorted(set(activity_names), key=lambda x:activity_names.index(x))

def compute_columns(activity_names:list):
    '''
        Compute the column names for the dataset based on activity names.
        Parameters:
            activity_names (list): List of activity names.
        Returns:
            list: List of column names including 'trace_id', activity names, and 'label'.
    '''
    # Start with 'trace_id' column which identifies each trace
    columns =['trace_id']
    # Add activity names of the log
    columns+=activity_names
    # Add 'label' column for ground truth
    columns.append('label')
    return columns

def boolean_encode(log:elem.EventLog, activity_names:list):
    '''
        Boolean encode the event log into a DataFrame.
        Parameters:
            log (elem.EventLog): The event log object.
            activity_names (list): List of activity names.
        Returns:
            pd.DataFrame: Boolean encoded DataFrame of the event log.
    '''
    encoded_log = []
    # Build column names
    columns = compute_columns(activity_names)
    for case in log:
        # Initialize the encoded row with trace_id
        encoded_row = [case.attributes["concept:name"]]
        # Initialize boolean indicators for each activity
        bool_events = [False]*len(activity_names)   
        for event in case:
            event_name = event["concept:name"]
            # If the event name is present in activity names, that means this activity occurred in the trace
            if event_name in activity_names:
                # Get the index of the activity name in order to update the corresponding boolean indicator
                activity_name_index = activity_names.index(event["concept:name"])
            bool_events[activity_name_index]=True
        # Append boolean indicators and label to the encoded row
        encoded_row += bool_events
        encoded_row.append(case.attributes["label"])
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

def extract_recommendations(tree, feature_names, class_values, prefix_set):
    '''
        Extract recommendations from a decision tree for given prefixes.
            Parameters:
                tree: The decision tree model.
                feature_names: List of feature names used in the model.
                class_values: Possible class values for the target variable.
                prefix_set: Set of prefixes to extract recommendations for.
            Returns:
                dict: A dictionary mapping prefixes to their recommendations.
    '''
    return 0

def evaluate_recommendations(test_set, recommendations):
    '''
        Evaluate the recommendations against the test set.
            Parameters:
                test_set: The test dataset.
                recommendations: The recommendations to evaluate.
            Returns:
                float: The evaluation score.
    '''
    return 0