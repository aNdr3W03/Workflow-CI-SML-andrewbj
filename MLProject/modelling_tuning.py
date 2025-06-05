import os
import json
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('modelling_tuning.py')

def mlflow_setup():
    try:
        if dagshub_username and dagshub_token:
            mlflow_url = f'https://dagshub.com/{dagshub_username}/{dagshub_repo_name}.mlflow'
            mlflow.set_tracking_uri(mlflow_url)
        else:
            raise ValueError('DagsHub Username and Token must set on the environment.')
        
        mlflow.set_experiment('Diabetes Prediction Tuning CI/CD')
        logger.info('MLflow setup for DagsHub completed.')
        
    except Exception as e:
        logger.exception(f'MLflow setup for DagsHub failed: {e}.')
        mlflow.set_tracking_uri('file:./mlruns ')
        mlflow.set_experiment('Diabetes Prediction Tuning CI/CD')
        logger.info('MLflow setup locally completed.')

def load_data(data_path='diabetes_processed.csv'):
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
        'accuracy': accuracy_score(y_test, y_pred),     # Overall correctness of the model
        'precision': precision_score(y_test, y_pred),   # Trustworthiness of "positive" predictions
        'recall': recall_score(y_test, y_pred),         # AKA sensitivity; How well the model identifies diabetics
        'f1_score': f1_score(y_test, y_pred),           # Balance between precision and recall
        'cm_true_negative': tn,
        'cm_false_positive': fp,
        'cm_false_negative': fn,
        'cm_true_positive': tp,
        'tnr': tn / (tn + fp) if (tn + fp) != 0 else 0, # AKA specificity; How well the model identifies non-diabetics
        'fnr': fn / (fn + tp) if (fn + tp) != 0 else 0, # AKA miss-rate; How well the model fails to identify diabetics (track underdiagnosis)
        'fpr': fp / (fp + tn) if (fp + tn) != 0 else 0, # AKA fall-out; How well the model fails to identifies non-diabetics as diabetics (track overdiagnosis)
    }

    logger.info(f'Model {model} evaluated. Accuracy: {metrics['accuracy']:.4f}.')

    return metrics, cm

def lr_model_tuning(X_train, X_test, y_train, y_test):
    logger.info('Starting Logistic Regression model tuning...')\
    
    param = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'max_iter': [100, 500, 1000]
    }

    lr = LogisticRegression(random_state=20250531)

    kf = KFold(n_splits=5, shuffle=True, random_state=20250531)

    gs = GridSearchCV(lr, param, scoring='accuracy', cv=kf, verbose=1, n_jobs=-1)

    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    logger.info(f'Logistic Regression best params: {best_params}')

    best_model = LogisticRegression(**best_params, random_state=20250531)
    best_model.fit(X_train, y_train)

    metrics, cm = model_evaluate(best_model, X_test, y_test)
    logger.info(f'Logistic Regression model accuracy: {metrics['accuracy']:.4f}')
    
    with mlflow.start_run(run_name='lr_tuned_run') as run:
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        logger.info('Param logged to MLflow.')
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        logger.info('Metrics logged to MLflow.')

        cm_df = pd.DataFrame(
            cm, index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df.T, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for lr Tuned', fontweight='bold', pad=10)
        plt.xlabel('Actual', fontweight='bold')
        plt.ylabel('Predicted', fontweight='bold')
        plt.tight_layout()
        cm_plot_path = 'models_tuned/lr_tuned_confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        logger.info('Confusion matrix logged as PNG to MLflow.')

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.coef_[0]
        }).sort_values(by='importance', ascending=False)

        importance_path = 'models_tuned/lr_tuned_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        for feature, importance_value in zip(X_train.columns, best_model.coef_[0]):
            mlflow.log_param(f'importance_{feature}', importance_value)
        logger.info('Feature importance logged to MLflow.')

        metrics_path = 'models_tuned/lr_tuned_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)
        logger.info('Metrics saved and logged to MLflow.')

        mlflow.sklearn.log_model(
            best_model, 'lr_tuned',
            input_example=X_train.head(),
            signature=infer_signature(X_train, best_model.predict(X_train))
        )

        model_path = 'models_tuned/lr_tuned_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info('Logistic Regression model logged to MLflow.')

        run_id = run.info.run_id
        logger.info('Logistic Regression model tuning completed.')

    return best_model, metrics, cm, run_id

def rf_model_tuning(X_train, X_test, y_train, y_test):
    logger.info('Starting Random Forest model tuning...')

    param = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=20250531)

    kf = KFold(n_splits=5, shuffle=True, random_state=20250531)

    gs = GridSearchCV(rf, param, scoring='accuracy', cv=kf, verbose=1, n_jobs=-1)

    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    logger.info(f'Random Forest best params: {best_params}')

    best_model = RandomForestClassifier(**best_params, random_state=20250531)
    best_model.fit(X_train, y_train)

    metrics, cm = model_evaluate(best_model, X_test, y_test)
    logger.info(f'Random Forest model accuracy: {metrics['accuracy']:.4f}')

    with mlflow.start_run(run_name='rf_tuned_run') as run:
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        logger.info('Param logged to MLflow.')
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        logger.info('Metrics logged to MLflow.')

        cm_df = pd.DataFrame(
            cm, index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df.T, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for rf Tuned', fontweight='bold', pad=10)
        plt.xlabel('Actual', fontweight='bold')
        plt.ylabel('Predicted', fontweight='bold')
        plt.tight_layout()
        cm_plot_path = 'models_tuned/rf_tuned_confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        logger.info('Confusion matrix logged as PNG to MLflow.')

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        importance_path = 'models_tuned/rf_tuned_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        for feature, importance_value in zip(X_train.columns, best_model.feature_importances_):
            mlflow.log_param(f'importance_{feature}', importance_value)
        logger.info('Feature importance logged to MLflow.')

        metrics_path = 'models_tuned/rf_tuned_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)
        logger.info('Metrics saved and logged to MLflow.')

        mlflow.sklearn.log_model(
            best_model, 'rf_tuned',
            input_example=X_train.head(),
            signature=infer_signature(X_train, best_model.predict(X_train))
        )

        model_path = 'models_tuned/rf_tuned_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info('Random Forest model logged to MLflow.')

        run_id = run.info.run_id
        logger.info('Random Forest model tuning completed.')

    return best_model, metrics, cm, run_id

def adaboost_model_tuning(X_train, X_test, y_train, y_test):
    logger.info('Starting AdaBoost model tuning...')

    param = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'estimator': [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(max_depth=3),
            DecisionTreeClassifier(max_depth=5)
        ]
    }

    ada = AdaBoostClassifier(random_state=20250531)

    kf = KFold(n_splits=5, shuffle=True, random_state=20250531)

    gs = GridSearchCV(ada, param, scoring='accuracy', cv=kf, verbose=1, n_jobs=-1)

    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    logger.info(f'AdaBoost best params: {best_params}')

    best_model = AdaBoostClassifier(**best_params, random_state=20250531)
    best_model.fit(X_train, y_train)

    metrics, cm = model_evaluate(best_model, X_test, y_test)
    logger.info(f'AdaBoost model accuracy: {metrics['accuracy']:.4f}')

    with mlflow.start_run(run_name='adaboost_tuned_run') as run:
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        logger.info('Param logged to MLflow.')
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        logger.info('Metrics logged to MLflow.')

        cm_df = pd.DataFrame(
            cm, index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df.T, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for adaboost Tuned', fontweight='bold', pad=10)
        plt.xlabel('Actual', fontweight='bold')
        plt.ylabel('Predicted', fontweight='bold')
        plt.tight_layout()
        cm_plot_path = 'models_tuned/adaboost_tuned_confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        logger.info('Confusion matrix logged as PNG to MLflow.')

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        importance_path = 'models_tuned/adaboost_tuned_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        for feature, importance_value in zip(X_train.columns, best_model.feature_importances_):
            mlflow.log_param(f'importance_{feature}', importance_value)
        logger.info('Feature importance logged to MLflow.')

        metrics_path = 'models_tuned/adaboost_tuned_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)
        logger.info('Metrics saved and logged to MLflow.')

        mlflow.sklearn.log_model(
            best_model, 'adaboost_tuned',
            input_example=X_train.head(),
            signature=infer_signature(X_train, best_model.predict(X_train))
        )

        model_path = 'models_tuned/adaboost_tuned_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info('AdaBoost model logged to MLflow.')

        run_id = run.info.run_id
        logger.info('AdaBoost model tuning completed.')

    return best_model, metrics, cm, run_id

def dt_model_tuning(X_train, X_test, y_train, y_test):
    logger.info('Starting Decision Tree model tuning...')

    param = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    dt = DecisionTreeClassifier(random_state=20250531)

    kf = KFold(n_splits=5, shuffle=True, random_state=20250531)

    gs = GridSearchCV(dt, param, scoring='accuracy', cv=kf, verbose=1, n_jobs=-1)

    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    logger.info(f'Decision Tree best params: {best_params}')

    best_model = DecisionTreeClassifier(**best_params, random_state=20250531)
    best_model.fit(X_train, y_train)

    metrics, cm = model_evaluate(best_model, X_test, y_test)
    logger.info(f'Decision Tree model accuracy: {metrics['accuracy']:.4f}')

    with mlflow.start_run(run_name='dt_tuned_run') as run:
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        logger.info('Param logged to MLflow.')
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        logger.info('Metrics logged to MLflow.')

        cm_df = pd.DataFrame(
            cm, index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df.T, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for dt Tuned', fontweight='bold', pad=10)
        plt.xlabel('Actual', fontweight='bold')
        plt.ylabel('Predicted', fontweight='bold')
        plt.tight_layout()
        cm_plot_path = 'models_tuned/dt_tuned_confusion_matrix.png'
        os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        logger.info('Confusion matrix logged as PNG to MLflow.')

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        importance_path = 'models_tuned/dt_tuned_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        for feature, importance_value in zip(X_train.columns, best_model.feature_importances_):
            mlflow.log_param(f'importance_{feature}', importance_value)
        logger.info('Feature importance logged to MLflow.')

        metrics_path = 'models_tuned/dt_tuned_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)
        logger.info('Metrics saved and logged to MLflow.')

        mlflow.sklearn.log_model(
            best_model, 'dt_tuned',
            input_example=X_train.head(),
            signature=infer_signature(X_train, best_model.predict(X_train))
        )

        model_path = 'models_tuned/dt_tuned_model.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info('Decision Tree model saved logged to MLflow.')

        run_id = run.info.run_id
        logger.info('Decision Tree model tuning completed.')

    return best_model, metrics, cm, run_id

def main():
    try:
        mlflow_setup()

        X_train, X_test, y_train, y_test = load_data()

        results = {}
        
        lr_model, lr_metrics, lr_cm, lr_run_id = lr_model_tuning(X_train, X_test, y_train, y_test)
        results['logistic_regression'] = {
            'model': lr_model,
            'metrics': lr_metrics,
            'cm': lr_cm,
            'run_id': lr_run_id
        }

        rf_model, rf_metrics, rf_cm, rf_run_id = rf_model_tuning(X_train, X_test, y_train, y_test)
        results['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics,
            'cm': rf_cm,
            'run_id': rf_run_id
        }

        adaboost_model, adaboost_metrics, adaboost_cm, adaboost_run_id = adaboost_model_tuning(X_train, X_test, y_train, y_test)
        results['adaptive_boosting'] = {
            'model': adaboost_model,
            'metrics': adaboost_metrics,
            'cm': adaboost_cm,
            'run_id': adaboost_run_id
        }

        dt_model, dt_metrics, dt_cm, dt_run_id = dt_model_tuning(X_train, X_test, y_train, y_test)
        results['decision_tree'] = {
            'model': dt_model,
            'metrics': dt_metrics,
            'cm': dt_cm,
            'run_id': dt_run_id
        }

        best_model_name = max(
            results.keys(),
            key=lambda k: (
                results[k]['metrics']['accuracy'],
                results[k]['metrics']['precision'],
                results[k]['metrics']['recall']
            )
        )
        best_model_info = results[best_model_name]

        logger.info(f'Best model: {best_model_name}')
        logger.info(f'Accuracy: {best_model_info['metrics']['accuracy']:.4f}')
        logger.info(f'Run ID: {best_model_info['run_id']}')

        run_id_path = 'models_tuned/best_tuned_model_run_id.txt'
        os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
        with open(run_id_path, 'w') as f:
            f.write(best_model_info['run_id'])

        best_model_path = f'models_tuned/{best_model_name}_tuned_model.pkl'
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        joblib.dump(best_model_info['model'], best_model_path)

        logger.info(f'Best tuned model saved to: {best_model_path}')
        logger.info('All models tuned and logged successfully.')

        return results
    except Exception as e:
        logger.exception(f'An error occurred during model tuning: {e}')
        raise

if __name__ == '__main__':
    main()