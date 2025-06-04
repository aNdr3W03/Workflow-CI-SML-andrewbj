import os
import json
import joblib
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from dotenv import load_dotenv
load_dotenv()

dagshub_username = os.environ.get('DAGSHUB_USERNAME')
dagshub_token = os.environ.get('DAGSHUB_TOKEN')
dagshub_repo_name = 'diabetes-prediction'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('modelling.py')

def mlflow_setup():
    try:
        if dagshub_username and dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

            mlflow_url = f'https://dagshub.com/{dagshub_username}/{dagshub_repo_name}.mlflow'
            mlflow.set_tracking_uri(mlflow_url)
        else:
            raise ValueError('DagsHub Username and Token must set on the environment.')
        
        mlflow.set_experiment('Diabetes Prediction')
        logger.info('MLflow setup for DagsHub completed.')
        
    except Exception as e:
        logger.error(f'MLflow setup for DagsHub failed: {str(e)}.')
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.set_experiment('Diabetes Prediction')
        logger.info('MLflow setup locally completed.')

def load_data(data_path='modelling/diabetes_processed.csv'):
    logger.info(f'Loading data from: {data_path}')
    df = pd.read_csv(data_path)
    
    X = df.drop(['diabetes'], axis=1)
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20250531, stratify=y
    )

    # Return the train-test split data
    logger.info(f'Data loaded and split. Train: {X_train.shape}, Test: {X_test.shape}')
    return X_train, X_test, y_train, y_test

def model_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'cm_true_negative': tn,
        'cm_false_positive': fp,
        'cm_false_negative': fn,
        'cm_true_positive': tp,
    }

    logger.info(f'Model {model} evaluated. Accuracy: {metrics['accuracy']:.4f}.')

    return metrics, cm

def model_train(X_train, X_test, y_train, y_test, model_name, params=None):
    logger.info(f'Model training started: {model_name}. Parameters: {params}')
    if params is None:
        params = {}

    # Initialize the model with parameters
    if model_name == 'lr':
        model = LogisticRegression(**params, random_state=20250531)
    elif model_name == 'rf':
        model = RandomForestClassifier(**params, random_state=20250531)
    elif model_name == 'adaboost':
        model = AdaBoostClassifier(**params, random_state=20250531)
    elif model_name == 'dt':
        model = DecisionTreeClassifier(**params, random_state=20250531)
    else:
        logger.error(f'Unsupported model: {model_name}')
        raise ValueError(f'Model {model_name} is not supported.')

    # Model training
    model.fit(X_train, y_train)
    logger.info('Model training completed.')

    # Model evaluation
    metrics, cm = model_evaluate(model, X_test, y_test)

    # Cross-validation accuracy
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    logger.info(f'Cross-validation accuracy: {cv_accuracy:.4f}')

    return model, metrics, cm, cv_accuracy

def mlflow_log(model, model_name, params, metrics, cv_accuracy, input_data, cm):
    logger.info('Logging to MLflow.')
    with mlflow.start_run(run_name=f'{model_name}_run') as run:
        for param, value in params.items():
            mlflow.log_param(param, value)
        logger.info('Param logged to MLflow.')
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        mlflow.log_metric('accuracy_crossval', cv_accuracy)
        logger.info('Metrics logged to MLflow.')

        cm_df = pd.DataFrame(
            cm, index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df.T, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}', fontweight='bold', pad=10)
        plt.xlabel('Actual', fontweight='bold')
        plt.ylabel('Predicted', fontweight='bold')
        plt.tight_layout()
        cm_plot_path = f'models/{model_name}_confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        logger.info('Confusion matrix logged as CSV and PNG to MLflow.')

        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': input_data.columns,
                'importance': model.feature_importances_,
            }).sort_values(by='importance', ascending=False)

            importance_path = f'models/{model_name}_feature_importance.csv'
            os.makedirs(os.path.dirname(importance_path), exist_ok=True)
            importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

            for feature, importance_value in zip(input_data.columns, model.feature_importances_):
                mlflow.log_param(f'importance_{feature}', importance_value)
            
            logger.info('Feature importance logged to MLflow.')

        metrics_path = f'models/{model_name}_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)
        logger.info('Metrics saved and logged to MLflow.')
        
        mlflow.sklearn.log_model(
            model,
            'model',
            input_example=input_data.head(),
            signature=infer_signature(input_data, model.predict(input_data))
        )

        model_path = f'models/{model_name}.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        logger.info('Model artifact saved and logged to MLflow.')
        
        return run.info.run_id

def main(args):
    try:
        mlflow_setup()

        X_train, X_test, y_train, y_test = load_data(args.data_path)
        
        model_params = {
            'lr': {'C': 1.0, 'max_iter': 100},
            'rf': {'n_estimators': 100, 'max_depth': 10},
            'adaboost': {'n_estimators': 50, 'learning_rate': 1.0},
            'dt': {'max_depth': 10}
        }

        if args.tuning:
            import modelling_tuning
            logger.info('Model hyperparameter tuning started...')

            results = modelling_tuning.main()
            best_model = max(
                results.keys(),
                key=lambda k: (
                    results[k]['metrics']['accuracy'],
                    results[k]['metrics']['precision'],
                    results[k]['metrics']['recall']
                )
            )
            best_run_id = results[best_model]['run_id']
            logger.info(f'Best tuned model: {best_model} with run ID {best_run_id}')

            return best_run_id
        else:
            model_name = args.model_name
            params = model_params.get(model_name, {})

            model, metrics, cm, cv_accuracy = model_train(
                X_train, X_test, y_train, y_test, model_name, params
            )

            run_id = mlflow_log(model, model_name, params, metrics, cv_accuracy, X_train, cm)

            logger.info(f'Model {model_name} trained and logged successfully.')
            logger.info(f'Run ID: {run_id}')
            logger.info(f'Accuracy: {metrics['accuracy']:.4f}')
            logger.info(f'Cross-validation accuracy: {cv_accuracy:.4f}')

            run_id_path = 'models/model_run_id.txt'
            os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
            with open(run_id_path, 'w') as f:
                f.write(run_id)
            logger.info('Run ID saved to file.')

            return run_id
    
    except Exception as e:
        logger.exception(f'An error occurred when training model: {e}')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and log a diabetes prediction model using MLflow.')

    parser.add_argument(
        '-d', '--data_path', type=str, default='modelling/diabetes_processed.csv',
        help='Path to the processed data file.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-m', '--model_name', type=str, choices=['lr', 'rf', 'adaboost', 'dt'],
        help='Name of the model to train. If specified, tuning must not be enabled.')
    group.add_argument(
        '-t', '--tuning', action='store_true',
        help='Enable hyperparameter tuning. If used, model name must not be specified.')

    args = parser.parse_args()
    
    main(args)