"""
K-Nearest Neighbors Module
Contains functions for training and evaluating KNN models
with Pipeline structure and feature selection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


class L1FeatureSelector(BaseEstimator, TransformerMixin):
    """L1-based feature selection using Logistic Regression"""
    
    def __init__(self, C=1.0, threshold=1e-5):
        self.C = C
        self.threshold = threshold
        self.selector = None
        self.support_ = None
        
    def fit(self, X, y):
        self.selector = LogisticRegression(
            penalty='l1', C=self.C, solver='liblinear',
            max_iter=1000, random_state=42
        )
        self.selector.fit(X, y)
        self.support_ = np.abs(self.selector.coef_[0]) > self.threshold
        return self
    
    def transform(self, X):
        return X[:, self.support_]
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.arange(np.sum(self.support_))
        return np.array(input_features)[self.support_]


class KBestFeatureSelector(BaseEstimator, TransformerMixin):
    """KBest feature selection wrapper"""
    
    def __init__(self, k=10):
        self.k = k
        self.selector = None
        
    def fit(self, X, y):
        self.selector = SelectKBest(f_classif, k=min(self.k, X.shape[1]))
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        return self.selector.get_feature_names_out(input_features)


class NoOpTransformer(BaseEstimator, TransformerMixin):
    """Pass-through transformer (no feature selection)"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.arange(X.shape[1]) if hasattr(self, 'X') else None
        return input_features


def knn_pipeline_analysis(
    X, 
    y, 
    dataset_name, 
    use_gridsearch=True,
    test_size=0.2, 
    random_state=42
):
    """
    Complete KNN analysis with proper Pipeline (NO DATA LEAKAGE).
    
    Pipeline: Imputer → FeatureSelector → Scaler → KNN
    
    Parameters
    ----------
    X : DataFrame
        Features
    y : Series
        Target variable
    dataset_name : str
        Dataset name
    use_gridsearch : bool
        Use GridSearchCV or simple training
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    
    Returns
    -------
    dict
        Complete results including metrics and predictions
    """
    
    print("\n" + "="*80)
    print(f"  KNN PIPELINE ANALYSIS: {dataset_name}")
    print("="*80)
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    # Convert to numpy
    X_np = X.values if hasattr(X, 'values') else X
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    
    print(f"\n   Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print(f"     Train: {train_counts.get(0, 0)} survived, {train_counts.get(1, 0)} died")
    print(f"     Test:  {test_counts.get(0, 0)} survived, {test_counts.get(1, 0)} died")
    
    # Build pipeline
    if use_gridsearch:
        print(f"\n   Running GridSearchCV with Pipeline...")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', NoOpTransformer()),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(algorithm='brute', n_jobs=1))
        ])
        
        param_grid = {
            'selector': [
                NoOpTransformer(),
                L1FeatureSelector(C=0.1),
                L1FeatureSelector(C=0.5),
                L1FeatureSelector(C=1.0),
                KBestFeatureSelector(k=5),
                KBestFeatureSelector(k=10),
                KBestFeatureSelector(k=15),
            ],
            'knn__n_neighbors': [3, 5, 7, 9, 11]
        }
        
        total_combinations = len(param_grid['selector']) * len(param_grid['knn__n_neighbors'])
        print(f"     Testing {total_combinations} combinations...")
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\n   GridSearch Complete!")
        print(f"\n   BEST PARAMETERS:")
        
        # Extract readable info
        selector_obj = best_params['selector']
        if isinstance(selector_obj, L1FeatureSelector):
            method = 'L1'
            detail = f"C={selector_obj.C}"
        elif isinstance(selector_obj, KBestFeatureSelector):
            method = 'KBest'
            detail = f"k={selector_obj.k}"
        else:
            method = 'None'
            detail = ''
        
        k = best_params['knn__n_neighbors']
        
        print(f"     k = {k}")
        print(f"     Method = {method}")
        if detail:
            print(f"     Params = {detail}")
        
        # Count selected features
        if hasattr(best_pipeline.named_steps['selector'], 'support_'):
            n_selected = np.sum(best_pipeline.named_steps['selector'].support_)
            print(f"     Features: {X.shape[1]} → {n_selected}")
        elif hasattr(best_pipeline.named_steps['selector'], 'selector'):
            if hasattr(best_pipeline.named_steps['selector'].selector, 'k'):
                n_selected = best_pipeline.named_steps['selector'].selector.k
                print(f"     Features: {X.shape[1]} → {n_selected}")
        
        print(f"     CV Accuracy: {best_score:.3f}")
        
        best_params_clean = {
            'k': k,
            'method': method,
            'detail': detail
        }
        
    else:
        print(f"\n   Training simple Pipeline (k=5, no selection)...")
        
        best_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', NoOpTransformer()),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, algorithm='brute', n_jobs=1))
        ])
        
        best_pipeline.fit(X_train, y_train)
        
        best_params_clean = {'k': 5, 'method': 'None', 'detail': ''}
        grid_search = None
    
    # Predictions
    print(f"\n   Making predictions...")
    
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n   Test Performance:")
    print(f"     Accuracy: {acc:.3f}, AUC: {roc_auc:.3f}, F1: {f1:.3f}")
    
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Survived', 'Died'],
                                zero_division=0))
    
    return {
        'name': dataset_name,
        'pipeline': best_pipeline,
        'best_params': best_params_clean,
        'grid_search': grid_search,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': {
            'accuracy': acc,
            'balanced_acc': bal_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        },
        'cm': confusion_matrix(y_test, y_pred),
        'train_counts': train_counts,
        'test_counts': test_counts
    }


def create_summary_table(all_results):
    """Create summary table for all KNN results"""
    print("\n" + "="*100)
    print("SUMMARY: KNN PIPELINE WITH GRIDSEARCH & FEATURE SELECTION")
    print("="*100)
    print(f"{'Dataset':<35} {'k':<5} {'Method':<10} {'Acc':<8} {'AUC':<8} {'F1':<8}")
    print("-"*100)
    
    for r in all_results:
        name = r['name']
        params = r['best_params']
        m = r['metrics']
        print(f"{name:<35} {params['k']:<5} {params['method']:<10} "
              f"{m['accuracy']:<8.3f} {m['roc_auc']:<8.3f} {m['f1']:<8.3f}")
    
    print("="*100)
    
    best_idx = np.argmax([r['metrics']['roc_auc'] for r in all_results])
    print(f"\n BEST: {all_results[best_idx]['name']}")
    print(f"   AUC: {all_results[best_idx]['metrics']['roc_auc']:.3f}")
    print(f"   Pipeline: k={all_results[best_idx]['best_params']['k']}, "
          f"{all_results[best_idx]['best_params']['method']}")


def run_knn_on_all_datasets(datasets_dict, y, test_size=0.2, random_state=42):
    """
    Run KNN pipeline analysis on all datasets.
    
    Parameters
    ----------
    datasets_dict : dict
        Dictionary of {name: DataFrame}
    y : Series
        Target variable
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns
    -------
    list
        List of result dictionaries for each dataset
    """
    
    print("\n" + "="*80)
    print("PIPELINE STRUCTURE")
    print("="*80)
    print("Pipeline([")
    print("    ('imputer', SimpleImputer(strategy='median')),")
    print("    ('selector', FeatureSelector()),  # L1, KBest, or None")
    print("    ('scaler', StandardScaler()),")
    print("    ('knn', KNeighborsClassifier())")
    print("])")
    print("\n NO DATA LEAKAGE - All preprocessing in Pipeline!")
    print("  GridSearch on full Pipeline")
    print("  Feature Selection (L1 + KBest)")
    print("="*80)
    
    all_results = []
    
    for dataset_name, X in datasets_dict.items():
        result = knn_pipeline_analysis(
            X=X,
            y=y,
            dataset_name=dataset_name,
            use_gridsearch=True,
            test_size=test_size,
            random_state=random_state
        )
        all_results.append(result)
    
    create_summary_table(all_results)
    
    return all_results


print(" KNN module loaded successfully!")
