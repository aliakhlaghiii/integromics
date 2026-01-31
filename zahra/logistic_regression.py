"""
Logistic Regression Module
Contains functions for training and evaluating Logistic Regression models
with multiple regularization types and feature selection methods
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# Global verbosity setting
VERBOSE = 1

def _vprint(level: int, msg: str):
    """Print message if verbosity level is sufficient"""
    if VERBOSE >= level:
        print(msg)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom feature selector for scikit-learn Pipeline.
    
    Supports 5 methods:
    1. 'none': Keep all features
    2. 'correlation': Remove highly correlated features
    3. 'kbest': Select k best features using f_classif
    4. 'rf': Select top features by RandomForest importance
    5. 'l1': Select features using L1-penalized LogisticRegression
    
    Parameters
    ----------
    method : str, default='none'
        Feature selection method
    correlation_threshold : float, default=0.9
        Threshold for correlation-based selection
    l1_C : float, default=1.0
        Regularization parameter for L1 selection
    kbest_k : int, default=10
        Number of features for KBest selection
    rf_n_features : int, default=10
        Number of features for RF selection
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        method: str = "none",
        correlation_threshold: float = 0.9,
        l1_C: float = 1.0,
        kbest_k: int = 10,
        rf_n_features: int = 10,
        random_state: int = 42
    ):
        self.method = method
        self.correlation_threshold = correlation_threshold
        self.l1_C = l1_C
        self.kbest_k = kbest_k
        self.rf_n_features = rf_n_features
        self.random_state = random_state
        self.selected_features_: Optional[List[str]] = None
    
    def fit(self, X, y=None):
        """Fit the feature selector to data"""
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # METHOD 1: No Selection
        if self.method == "none":
            self.selected_features_ = X_df.columns.tolist()
            return self
        
        # METHOD 2: Correlation-based Selection
        if self.method == "correlation":
            corr = X_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [c for c in upper.columns 
                      if (upper[c] > self.correlation_threshold).any()]
            selected = [c for c in X_df.columns if c not in to_drop]
            self.selected_features_ = selected if len(selected) > 0 else X_df.columns.tolist()
            return self
        
        # METHOD 3: K-Best Selection
        if self.method == "kbest":
            k = min(int(self.kbest_k), X_df.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X_df, y)
            self.selected_features_ = X_df.columns[selector.get_support()].tolist()
            return self
        
        # METHOD 4: RandomForest Feature Importance
        if self.method == "rf":
            n = min(int(self.rf_n_features), X_df.shape[1])
            rf = RandomForestClassifier(
                n_estimators=200, 
                random_state=self.random_state, 
                n_jobs=1
            )
            rf.fit(X_df, y)
            imp = rf.feature_importances_
            idx = np.argsort(imp)[::-1][:n]
            self.selected_features_ = X_df.columns[idx].tolist()
            return self
        
        # METHOD 5: L1-based Selection
        if self.method == "l1":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_df)
            
            lr = LogisticRegression(
                penalty="l1",
                C=self.l1_C,
                solver="liblinear",
                max_iter=2000,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=1
            )
            lr.fit(X_scaled, y)
            
            coef = lr.coef_.ravel()
            mask = (coef != 0)
            self.selected_features_ = X_df.columns[mask].tolist()
            
            # FALLBACK: If all features zeroed, keep top 10
            if len(self.selected_features_) == 0:
                n_keep = min(10, X_df.shape[1])
                top_idx = np.argsort(np.abs(coef))[::-1][:n_keep]
                self.selected_features_ = X_df.columns[top_idx].tolist()
            
            return self
        
        raise ValueError(f"Unknown method: {self.method}")
    
    def transform(self, X):
        """Transform data by selecting only fitted features"""
        if self.selected_features_ is None:
            raise RuntimeError("FeatureSelector not fitted. Call fit() first.")
        
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return X_df[self.selected_features_].copy()


def logistic_with_gridsearch(
    X: pd.DataFrame, 
    y: pd.Series, 
    dataset_name: str,
    fs_method: str = "none", 
    penalty: str = "l1",
    use_gridsearch: bool = True, 
    test_size: float = 0.2,
    random_state: int = 42, 
    scoring_main: str = "roc_auc"
) -> Dict[str, Any]:
    """
    Train Logistic Regression with GridSearch.
    
    Parameters
    ----------
    X : DataFrame
        Features
    y : Series
        Target variable
    dataset_name : str
        Name of dataset
    fs_method : str
        Feature selection method ('none', 'l1', 'kbest', 'rf', 'correlation')
    penalty : str
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    use_gridsearch : bool
        Whether to use GridSearchCV
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    scoring_main : str
        Scoring metric for GridSearch
        
    Returns
    -------
    dict
        Results including metrics, predictions, and model info
    """
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    _vprint(1, f"   Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Feature selector
    selector = FeatureSelector(method=fs_method, random_state=random_state)
    
    # Configure logistic regression based on penalty
    if penalty == "none":
        lr = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=2000,
            random_state=random_state, class_weight="balanced", n_jobs=1
        )
        param_grid = {}
    
    elif penalty == "l1":
        lr = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000,
            random_state=random_state, class_weight="balanced", n_jobs=1
        )
        if fs_method == "l1":
            param_grid = {
                "model__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "fs__l1_C": [0.5, 1.0, 2.0, 5.0, 10.0]
            }
        else:
            param_grid = {"model__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    
    elif penalty == "l2":
        lr = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000,
            random_state=random_state, class_weight="balanced", n_jobs=1
        )
        if fs_method == "l1":
            param_grid = {
                "model__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "fs__l1_C": [0.5, 1.0, 2.0, 5.0, 10.0]
            }
        else:
            param_grid = {"model__C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    
    elif penalty == "elasticnet":
        lr = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=3000,
            random_state=random_state, class_weight="balanced", n_jobs=1
        )
        param_grid = {
            "model__C": [0.01, 0.1, 0.5, 1.0, 2.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    else:
        raise ValueError(f"Unknown penalty: {penalty}")
    
    # Build pipeline
    pipe = Pipeline([
        ("fs", selector),
        ("scaler", StandardScaler()),
        ("model", lr)
    ])
    
    # GridSearch or simple training
    grid_results_df = None
    best_params = {}
    best_cv_score = None
    
    if use_gridsearch and len(param_grid) > 0:
        _vprint(1, f"   Running GridSearchCV...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            estimator=pipe, param_grid=param_grid, cv=cv,
            scoring=scoring_main, n_jobs=1, verbose=0, return_train_score=True
        )
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_
        best_params = gs.best_params_
        best_cv_score = gs.best_score_
        grid_results_df = pd.DataFrame(gs.cv_results_).sort_values("mean_test_score", ascending=False)
        _vprint(1, f"   Best CV {scoring_main}: {best_cv_score:.3f}")
    else:
        _vprint(1, f"   Training without GridSearch...")
        best_pipe = pipe.fit(X_train, y_train)
    
    # Predictions
    y_test_pred = best_pipe.predict(X_test)
    y_test_proba = best_pipe.predict_proba(X_test)[:, 1]
    
    # Metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # Feature counts
    fs_step = best_pipe.named_steps["fs"]
    selected_features_after_fs = fs_step.selected_features_
    n_after_fs = len(selected_features_after_fs)
    
    lr_step = best_pipe.named_steps["model"]
    coefficients = lr_step.coef_[0]
    
    if penalty in ["l1", "elasticnet"]:
        nonzero_mask = (coefficients != 0)
        n_selected_final = int(nonzero_mask.sum())
    else:
        n_selected_final = len(coefficients)
    
    _vprint(1, f"   Test: Acc={test_accuracy:.3f}, AUC={test_auc:.3f}, F1={test_f1:.3f}")
    _vprint(1, f"   Features: {X.shape[1]} → {n_after_fs} → {n_selected_final}")
    
    return {
        "dataset_name": dataset_name,
        "fs_method": fs_method,
        "penalty": penalty,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "n_features": X.shape[1],
        "n_features_after_fs": n_after_fs,
        "n_selected_features": n_selected_final,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "test_f1": test_f1,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
        "grid_results_df": grid_results_df,
        "pipeline": best_pipe
    }


def logistic_comprehensive_analysis(
    datasets: Dict[str, tuple],
    y: pd.Series,
    penalties: List[str] = None,
    fs_methods: List[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run comprehensive Logistic Regression analysis across all combinations.
    
    Tests ALL combinations of:
    - Penalties: none, L1, L2, ElasticNet
    - Datasets: X0, X1, X2, X3, ...
    - Feature Selection methods: none, L1, correlation, KBest, RF
    
    Parameters
    ----------
    datasets : dict
        Dictionary where key=dataset name, value=(DataFrame, description)
    y : Series
        Target vector
    penalties : list of str, optional
        Penalties to test. Default: ['none', 'l1', 'l2', 'elasticnet']
    fs_methods : list of str, optional
        FS methods. Default: ['none', 'l1']
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    
    Returns
    -------
    dict
        Complete results including summary DataFrame and best models
    """
    
    # Set defaults
    if penalties is None:
        penalties = ['none', 'l1', 'l2', 'elasticnet']
    
    if fs_methods is None:
        fs_methods = ['none', 'l1']
    
    all_results = {}
    
    # Print header
    print("\n" + "="*100)
    print("COMPREHENSIVE LOGISTIC REGRESSION ANALYSIS")
    print("="*100)
    print(f"  Testing {len(penalties)} penalties × {len(datasets)} datasets × {len(fs_methods)} FS methods")
    print(f"  Total: {len(penalties) * len(datasets) * len(fs_methods)} combinations")
    print("="*100)
    
    # Test all combinations
    for dataset_name, dataset_value in datasets.items():
        # Handle both tuple format and direct DataFrame format
        if isinstance(dataset_value, tuple):
            X_data, description = dataset_value
        else:
            X_data = dataset_value
            description = dataset_name
        for penalty in penalties:
            for fs_method in fs_methods:
                
                # Skip: No regularization + L1 FS doesn't make sense
                if penalty == 'none' and fs_method == 'l1':
                    continue
                
                combo_name = f"{dataset_name}_{penalty}_{fs_method}"
                full_name = f"{dataset_name}: {description} ({penalty.upper()}, {fs_method.upper()} FS)"
                
                print(f"\n{'='*100}")
                print(f"  {full_name}")
                print(f"{'='*100}")
                print(f"  Features: {X_data.shape[1]}")
                
                try:
                    results = logistic_with_gridsearch(
                        X=X_data, 
                        y=y, 
                        dataset_name=full_name,
                        fs_method=fs_method,
                        penalty=penalty,
                        use_gridsearch=True,
                        test_size=test_size,
                        random_state=random_state,
                        scoring_main='roc_auc'
                    )
                    
                    all_results[combo_name] = results
                    
                except Exception as e:
                    print(f"   Error: {e}")
                    continue
    
    # Create summary table
    print("\n\n" + "="*100)
    print("SUMMARY: ALL PENALTIES × ALL DATASETS")
    print("="*100)
    
    summary_data = []
    for combo_name, res in all_results.items():
        parts = combo_name.split('_')
        dataset = parts[0]
        penalty = parts[1] if len(parts) > 1 else 'unknown'
        fs = parts[2] if len(parts) > 2 else 'unknown'
        
        best_C = res['best_params'].get('model__C', 'N/A')
        
        summary_data.append({
            'Dataset': dataset,
            'Penalty': penalty,
            'FS': fs,
            'C': best_C,
            'Acc': res['test_accuracy'],
            'AUC': res['test_auc'],
            'F1': res['test_f1'],
            'Features': f"{res['n_features']}→{res['n_selected_features']}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary table
    print("\n" + "-"*100)
    print(f"{'Dataset':<12} {'Penalty':<12} {'FS':<8} {'C':<8} {'Acc':<8} {'AUC':<8} {'F1':<8} {'Features':<15}")
    print("-"*100)
    
    for dataset in datasets.keys():
        dataset_results = summary_df[summary_df['Dataset'] == dataset]
        
        for _, row in dataset_results.iterrows():
            C_str = f"{row['C']:.2f}" if isinstance(row['C'], (int, float)) else str(row['C'])
            
            print(f"{row['Dataset']:<12} {row['Penalty']:<12} {row['FS']:<8} "
                  f"{C_str:<8} {row['Acc']:<8.3f} {row['AUC']:<8.3f} "
                  f"{row['F1']:<8.3f} {row['Features']:<15}")
        
        print("-"*100)
    
    # Find best overall
    best_combo = max(all_results.items(), key=lambda x: x[1]['test_auc'])
    best_name, best_res = best_combo
    
    print(f"\n BEST OVERALL: {best_res['dataset_name']}")
    print(f"   AUC: {best_res['test_auc']:.3f}")
    print(f"   Accuracy: {best_res['test_accuracy']:.3f}")
    print(f"   F1: {best_res['test_f1']:.3f}")
    
    # Best per dataset
    print(f"\n BEST PER DATASET:")
    for dataset in datasets.keys():
        dataset_combos = {k: v for k, v in all_results.items() 
                         if k.startswith(dataset)}
        
        if dataset_combos:
            best_for_dataset = max(dataset_combos.items(), 
                                  key=lambda x: x[1]['test_auc'])
            res = best_for_dataset[1]
            
            parts = best_for_dataset[0].split('_')
            penalty = parts[1] if len(parts) > 1 else '?'
            fs = parts[2] if len(parts) > 2 else '?'
            C = res['best_params'].get('model__C', 'N/A')
            
            print(f"   {dataset}: AUC={res['test_auc']:.3f} "
                  f"({penalty.upper()}, {fs.upper()} FS, C={C})")
    
    return {
        'all_results': all_results,
        'summary_df': summary_df,
        'best_overall': best_res,
        'best_combo_name': best_name
    }


def print_detailed_analysis(all_results: dict, datasets: dict):
    """Print detailed analysis per dataset with classification reports"""
    
    print("\n" + "="*100)
    print("DETAILED ANALYSIS PER DATASET")
    print("="*100)
    
    for ds_name in datasets.keys():
        ds_results = {k: v for k, v in all_results.items() 
                      if k.startswith(ds_name)}
        
        if not ds_results:
            continue
        
        best_combo = max(ds_results.items(), key=lambda x: x[1]['test_auc'])
        best_name, best_res = best_combo
        
        parts = best_name.split('_')
        penalty = parts[1] if len(parts) > 1 else '?'
        fs = parts[2] if len(parts) > 2 else '?'
        
        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name} - {datasets[ds_name][1]}")
        print(f"{'='*100}")
        
        print(f"\n BEST MODEL:")
        print(f"   Penalty: {penalty.upper()}")
        print(f"   FS: {fs.upper()}")
        print(f"   AUC: {best_res['test_auc']:.3f}")
        print(f"   Accuracy: {best_res['test_accuracy']:.3f}")
        print(f"   F1: {best_res['test_f1']:.3f}")
        print(f"   Features: {best_res['n_features']} → {best_res['n_selected_features']}")
        
        print(f"\n CLASSIFICATION REPORT:")
        print(classification_report(
            best_res['y_test'], 
            best_res['y_test_pred'], 
            target_names=['Survived', 'Died'],
            zero_division=0
        ))
        
        cm = confusion_matrix(best_res['y_test'], best_res['y_test_pred'])
        print(f"\n CONFUSION MATRIX:")
        print(f"              Predicted")
        print(f"              Survived  Died")
        print(f"Actual Survived    {cm[0,0]:3d}     {cm[0,1]:3d}")
        print(f"       Died        {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    print("\n" + "="*100)
    print("DETAILED ANALYSIS COMPLETE")
    print("="*100)


print(" Logistic Regression module loaded successfully!")