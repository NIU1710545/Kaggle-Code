import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc







def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, method='grid', scoring='accuracy'):
    """
    Realitza cerca d'hiperparàmetres
    """
    if method == 'grid':
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            error_score='raise'
        )
    elif method == 'random':
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,  # Nombre d'iteracions per a cerca aleatòria
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    
    search.fit(X_train, y_train)
    
    print(f"Millors paràmetres: {search.best_params_}")
    print(f"Millor score (CV) [{scoring}]: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search

def evaluate_model(model, X_test, y_test, model_name=""):
    """
    Avaluació completa d'un model
    """
    print(f"\n{'='*60}")
    print(f"AVALUACIÓ DEL MODEL: {model_name}")
    print(f"{'='*60}")
    
    # Prediccions
    y_pred = model.predict(X_test)
    
    # IMPORTANT: Verificar els valors de y_test
    print(f"Valors únics a y_test: {np.unique(y_test)}")
    
    # Mètriques bàsiques - especificant pos_label per a classificació binària
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label=1),  # IMPORTANT
        'Recall': recall_score(y_test, y_pred, pos_label=1),        # IMPORTANT
        'F1-Score': f1_score(y_test, y_pred, pos_label=1)           # IMPORTANT
    }
    
    # Probabilitats (si el model les suporta)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilitat de la classe positiva (1)
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba)
        
        # Calcular corba ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)  # IMPORTANT
        roc_auc = auc(fpr, tpr)
        
        # Gràfic ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Corba ROC - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Imprimir mètriques
    print(f"\n📊 MÈTRIQUES D'AVALUACIÓ:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Matriu de confusió
    print(f"\n📊 Matriu de Confusió:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Gràfic de matriu de confusió
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Team 1 (0)', 'Team 2 (1)'], 
                yticklabels=['Team 1 (0)', 'Team 2 (1)'])
    plt.title(f'Matriu de Confusió - {model_name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predita')
    plt.show()
    
    # Informe de classificació
    print(f"\n📋 Informe de Classificació:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Team 1 (0)', 'Team 2 (1)'],
                                zero_division=0))
    
    # Gràfic addicional: Distribució de probabilitats (si existeix)
    if hasattr(model, 'predict_proba') and y_proba is not None:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_proba[y_test == 0], bins=30, alpha=0.7, label='Team 1 (0)', color='blue')
        plt.hist(y_proba[y_test == 1], bins=30, alpha=0.7, label='Team 2 (1)', color='red')
        plt.xlabel('Probabilitat de guanyar Team 2')
        plt.ylabel('Freqüència')
        plt.title('Distribució de Probabilitats')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        thresholds = np.arange(0, 1.01, 0.05)
        accuracies = []
        for thresh in thresholds:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            accuracies.append(accuracy_score(y_test, y_pred_thresh))
        
        plt.plot(thresholds, accuracies, 'o-', linewidth=2)
        plt.xlabel('Llindar de Decisió')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Llindar')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return metrics, cm

def cross_validation_analysis(model, X, y, cv=5, scoring='accuracy'):
    """
    Anàlisi exhaustiva de validació creuada amb múltiples mètriques
    
    Per a datasets balanceats com LoL:
    - Accuracy: OK per datasets balanceats
    - F1-Score: Millor (balanç Precision-Recall)
    - ROC-AUC: Ignora umbral de decisió
    - Precision/Recall: Per detectar desbalances
    """
    from sklearn.model_selection import cross_val_score, cross_validate
    
    print(f"\n{'='*60}")
    print(f"VALIDACIÓ CREUADA EXHAUSTIVA (CV={cv})")
    print(f"{'='*60}")
    
    # Calcular múltiples mètriques simultàniament
    scoring_dict = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(model, X, y, cv=cv, 
                               scoring=scoring_dict,
                               n_jobs=-1,
                               return_train_score=True)
    
    # Mostrar resultats per cada mètrica
    print(f"\n📊 RESULTATS PER CADA MÈTRICA:\n")
    
    metrics_summary = {}
    for metric_name in scoring_dict.keys():
        test_scores = cv_results[f'test_{metric_name}']
        train_scores = cv_results[f'train_{metric_name}']
        
        metrics_summary[metric_name] = {
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std(),
            'test_scores': test_scores
        }
        
        print(f"  {metric_name.upper():12} (Test):")
        print(f"    • Scores per fold: {[f'{s:.4f}' for s in test_scores]}")
        print(f"    • Mitjana: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
        print(f"    • IC 95%: [{test_scores.mean() - 1.96*test_scores.std():.4f}, "
              f"{test_scores.mean() + 1.96*test_scores.std():.4f}]")
        print(f"    • Train vs Test gap: {(train_scores.mean() - test_scores.mean()):.4f} "
              f"({'OVERFITTING' if (train_scores.mean() - test_scores.mean()) > 0.05 else 'OK'})")
        print()
    
    # Detectar problemes
    print(f"  🔍 DIAGNÒSTIC:\n")
    
    # Verificar si alguna mètrica és molt diferent
    f1_score_val = metrics_summary['f1']['test_mean']
    accuracy_val = metrics_summary['accuracy']['test_mean']
    
    if abs(f1_score_val - accuracy_val) > 0.05:
        print(f"    ⚠️  ALERTA: F1-Score ({f1_score_val:.4f}) diferent d'Accuracy ({accuracy_val:.4f})")
        print(f"       Pot indicar desbalance o problema amb la distribució de classes\n")
    
    precision_val = metrics_summary['precision']['test_mean']
    recall_val = metrics_summary['recall']['test_mean']
    
    if abs(precision_val - recall_val) > 0.05:
        print(f"    ⚠️  Precision ({precision_val:.4f}) vs Recall ({recall_val:.4f}) són molt diferents")
        if precision_val > recall_val:
            print(f"       → Model és més conservador (menys falsos positius)\n")
        else:
            print(f"       → Model és més agressiu (menys falsos negatius)\n")
    
    # Verificar overfitting
    max_gap = max([metrics_summary[m]['train_mean'] - metrics_summary[m]['test_mean'] 
                   for m in metrics_summary.keys()])
    
    if max_gap > 0.05:
        print(f"    ⚠️  Possible OVERFITTING: Gap train-test = {max_gap:.4f}\n")
    else:
        print(f"    ✅ No hi ha signe d'overfitting\n")
    
    # Gràfic comparatiu de totes les mètriques
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Totes les mètriques (mitjanes)
    plt.subplot(1, 2, 1)
    metrics_names = list(metrics_summary.keys())
    test_means = [metrics_summary[m]['test_mean'] for m in metrics_names]
    test_stds = [metrics_summary[m]['test_std'] for m in metrics_names]
    
    plt.bar(range(len(metrics_names)), test_means, yerr=test_stds, 
            capsize=5, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=2)
    plt.xticks(range(len(metrics_names)), [m.capitalize() for m in metrics_names], rotation=45)
    plt.ylabel('Score')
    plt.title(f'Comparació de Mètriques (CV={cv})')
    plt.ylim([0.90, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Afegir valors sobre les barres
    for i, (mean, std) in enumerate(zip(test_means, test_stds)):
        plt.text(i, mean + std + 0.002, f'{mean:.4f}', ha='center', fontsize=9, fontweight='bold')
    
    # Subplot 2: Evolució per fold (accuracy, f1, roc_auc)
    plt.subplot(1, 2, 2)
    folds = range(1, cv + 1)
    
    for metric in ['accuracy', 'f1', 'roc_auc']:
        scores = metrics_summary[metric]['test_scores']
        plt.plot(folds, scores, 'o-', label=metric.capitalize(), linewidth=2, markersize=8)
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Evolució per Fold (Mètriques principals)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.90, 1.0])
    
    plt.tight_layout()
    plt.show()
    
    # Tornar el score principal per compatibilitat
    return metrics_summary[scoring]['test_scores']

def get_param_grids():
    """
    Retorna els espais de paràmetres per a cada model
    """
    param_grids = {}
    
    # Regressió Logística - CORREGIT
    param_grids['logistic'] = [
        {
            'penalty': ['l2'],
            'solver': ['lbfgs', 'newton-cg', 'saga', 'liblinear'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [1000, 2000]
        },
        {
            'penalty': [None],
            'solver': ['lbfgs', 'newton-cg'],
            'max_iter': [1000, 2000]
        }
    ]
    
    # Random Forest
    param_grids['random_forest'] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Gradient Boosting
    param_grids['gradient_boosting'] = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # SVM
    param_grids['svm'] = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'degree': [2, 3, 4]  # Només per a kernel 'poly'
    }
    
    # XGBoost
    param_grids['xgboost'] = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # LightGBM
    param_grids['lightgbm'] = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # CatBoost
    param_grids['catboost'] = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5, 7]
    }
    
    return param_grids

def compare_models(results_dict, metric='Accuracy'):
    """
    Compara diferents models amb gràfics
    """
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(12, 6))
    
    # Gràfic de barres
    plt.subplot(1, 2, 1)
    bars = plt.barh(models, scores, color='skyblue')
    plt.xlabel(metric)
    plt.title(f'Comparació de Models per {metric}')
    
    # Afegir valors a les barres
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center')
    
    # Gràfic de punts
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(models)), scores, s=100, c='red', alpha=0.7)
    plt.xticks(range(len(models)), models, rotation=45)
    plt.ylabel(metric)
    plt.title('Distribució de Scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
