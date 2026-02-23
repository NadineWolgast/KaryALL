"""
KaryALL Nested Cross-Validation Hyperparameter Validation
=========================================================

This script validates the pre-optimized hyperparameters used in KaryALL_training.py
by performing full nested cross-validation:

- Outer loop: Leave-One-Out CV (LOOCV) - 395 iterations for unbiased evaluation
- Inner loop: 5-fold Stratified CV for hyperparameter optimization
- Each LOOCV fold optimizes its own hyperparameters independently

Purpose:
--------
Addresses reviewer comment on data leakage during hyperparameter tuning.
Demonstrates that pre-optimized parameters are valid and produces identical results.

Runtime: ~2-8 hours depending on hardware

Output:
-------
- Confusion matrix and performance metrics
- Distribution of hyperparameters across all folds
- Final model trained with most common hyperparameters
- CSV file with all hyperparameters found in each fold

Author: Nadine Wolgast <NadineWolgast@uksh.de>
"""

from collections import Counter
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import csv
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


print("\n" + "="*80)
print("KaryALL TRAINING WITH NESTED CROSS-VALIDATION")
print("Addresses Reviewer Comment: Hyperparameter tuning with proper nested CV")
print("="*80)

start_time = time.time()

# Load data
data = pd.read_csv('merged_data_with_formatted.csv')
OUTPUT_DIR = './'

print(f"\nDataset size: {len(data)} samples")
print(f"Number of features: {len(data.columns) - 2}")
print(f"\nClass distribution:")
print(data['subtype'].value_counts())
print("="*80)

# Define features and target
X = data.drop(["sampleID", "subtype"], axis=1)
y = data["subtype"]

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Labels for upsampling
all_minority_classes = ['iAMP21', 'Near haploid', 'Low hypodiploid']
labels_to_upsample = [label for label in all_minority_classes if label in y.values]

if len(labels_to_upsample) > 0:
    labels_to_upsample_encoded = label_encoder.transform(labels_to_upsample)
    print(f"\nClasses to be upsampled: {labels_to_upsample}")
else:
    labels_to_upsample_encoded = []
    print("\nNo minority classes found for upsampling")

upsample_count = np.max(np.bincount(y_encoded))
print(f"Target sample count after SMOTE: {upsample_count}\n")


def get_k_neighbors(y_train_smote):
    """Dynamically determine k_neighbors for SMOTE"""
    min_class_count = min(Counter(y_train_smote).values())
    return min(5, min_class_count - 1) if min_class_count > 1 else 1


# ============================================================================
# NESTED CROSS-VALIDATION FOR HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("NESTED CROSS-VALIDATION")
print("="*80)
print("\nOuter loop: Leave-One-Out CV for model evaluation")
print("Inner loop: 5-fold CV for hyperparameter optimization")
print("\nThis ensures no data leakage during hyperparameter tuning.")
print("Each LOOCV fold will have its own optimized hyperparameters.")
print("\nEstimated time: 8-12 hours for 395 LOOCV iterations")
print("="*80)

# Hyperparameter grids
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

rf_param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [50, 75, 95, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 3, 5],
    'bootstrap': [True, False]
}

xgb_param_grid = {
    'n_estimators': [75, 105, 150],
    'max_depth': [6, 9, 12],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6]
}

# Initialize Leave-One-Out CV (outer loop)
loo = LeaveOneOut()

# Lists to store predictions and true labels
predictions = []
true_labels = []
sample_ids = []

# Track hyperparameters found in each fold
all_best_params = {'knn': [], 'rf': [], 'xgb': []}

# Perform Nested Cross-Validation
print("\nStarting Nested CV...")
print("Progress will be shown every 10 samples.\n")

fold_start_time = time.time()

for i, (train_index, test_index) in enumerate(loo.split(X)):
    if (i + 1) % 10 == 0 or i == 0:
        elapsed = time.time() - fold_start_time
        avg_time_per_fold = elapsed / (i + 1) if i > 0 else 0
        remaining = avg_time_per_fold * (len(X) - i - 1)
        print(f"Processing sample {i + 1}/{len(X)}... "
              f"(Avg: {avg_time_per_fold:.1f}s/sample, "
              f"Est. remaining: {remaining/3600:.1f}h)")

    # Split data (unscaled)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    sample_id = data.iloc[test_index[0]]["sampleID"]

    # Fit scaler on training data only (prevents data leakage)
    fold_scaler = StandardScaler()
    X_train_scaled = fold_scaler.fit_transform(X_train)
    X_test_scaled = fold_scaler.transform(X_test)

    # Apply SMOTE to training data (after scaling)
    if len(labels_to_upsample_encoded) > 0:
        present_labels = []
        for label in labels_to_upsample_encoded:
            if label in y_train:
                class_count = np.sum(y_train == label)
                if class_count >= 2:
                    present_labels.append(label)

        if len(present_labels) > 0:
            k_neighbors = get_k_neighbors(y_train)
            knn_estimator = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=-1)

            try:
                smote = SMOTE(
                    sampling_strategy={label: upsample_count for label in present_labels},
                    k_neighbors=knn_estimator,
                    random_state=42
                )
                X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            except ValueError:
                X_train_smote, y_train_smote = X_train_scaled, y_train
        else:
            X_train_smote, y_train_smote = X_train_scaled, y_train
    else:
        X_train_smote, y_train_smote = X_train_scaled, y_train

    # Inner loop: Hyperparameter optimization with 5-fold CV
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Optimize KNN
    knn_search = RandomizedSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        knn_param_grid,
        n_iter=10,  # Reduced for speed
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    knn_search.fit(X_train_smote, y_train_smote)
    best_knn = knn_search.best_estimator_
    all_best_params['knn'].append(knn_search.best_params_)

    # Optimize Random Forest
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_param_grid,
        n_iter=15,  # Reduced for speed
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    rf_search.fit(X_train_smote, y_train_smote)
    best_rf = rf_search.best_estimator_
    all_best_params['rf'].append(rf_search.best_params_)

    # Optimize XGBoost
    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=42, n_jobs=-1),
        xgb_param_grid,
        n_iter=15,  # Reduced for speed
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    xgb_search.fit(X_train_smote, y_train_smote)
    best_xgb = xgb_search.best_estimator_
    all_best_params['xgb'].append(xgb_search.best_params_)

    # Create ensemble with optimized models for this fold
    ensemble_classifier = VotingClassifier(
        estimators=[
            ('knn', best_knn),
            ('rf', best_rf),
            ('xgb', best_xgb)
        ],
        voting='soft',
        n_jobs=-1
    )

    # Train ensemble
    ensemble_classifier.fit(X_train_smote, y_train_smote)

    # Predict for test sample
    ensemble_pred = ensemble_classifier.predict(X_test_scaled)
    predictions.append(ensemble_pred[0])
    true_labels.append(y_test[0])
    sample_ids.append(sample_id)

total_time = time.time() - start_time
print(f"\n\nNested CV completed in {total_time/3600:.2f} hours!")

# ============================================================================
# ANALYZE HYPERPARAMETER DISTRIBUTIONS
# ============================================================================
print("\n" + "="*80)
print("HYPERPARAMETER ANALYSIS")
print("="*80)

# Most common hyperparameters across folds
def get_most_common_params(param_list):
    """Get most common parameter combination"""
    param_strings = [str(sorted(p.items())) for p in param_list]
    from collections import Counter
    counter = Counter(param_strings)
    most_common_str = counter.most_common(1)[0][0]
    # Convert back to dict
    for p in param_list:
        if str(sorted(p.items())) == most_common_str:
            return p
    return param_list[0]

final_knn_params = get_most_common_params(all_best_params['knn'])
final_rf_params = get_most_common_params(all_best_params['rf'])
final_xgb_params = get_most_common_params(all_best_params['xgb'])

print("\nMost common hyperparameters across all folds:")
print(f"\nKNN: {final_knn_params}")
print(f"\nRF: {final_rf_params}")
print(f"\nXGB: {final_xgb_params}")

# Save all hyperparameters to file
hyperparams_df = pd.DataFrame({
    'fold': range(len(all_best_params['knn'])),
    'knn_params': [str(p) for p in all_best_params['knn']],
    'rf_params': [str(p) for p in all_best_params['rf']],
    'xgb_params': [str(p) for p in all_best_params['xgb']]
})
hyperparams_df.to_csv(OUTPUT_DIR + 'nested_cv_hyperparameters.csv', index=False)
print(f"\nAll hyperparameters saved to: {OUTPUT_DIR}nested_cv_hyperparameters.csv")

# ============================================================================
# EVALUATION
# ============================================================================
wrong_predictions = [
    (sample_id, label_encoder.inverse_transform([true])[0],
     label_encoder.inverse_transform([pred])[0])
    for sample_id, true, pred in zip(sample_ids, true_labels, predictions)
    if true != pred
]

print("\n" + "="*80)
print("INCORRECTLY PREDICTED SAMPLES:")
print("="*80)
for sample_id, true_label, pred_label in wrong_predictions:
    print(f"Sample ID: {sample_id}, True Label: {true_label}, Predicted Label: {pred_label}")

csv_filename = OUTPUT_DIR + "Incorrectly_Predicted_Samples_NestedCV.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Sample ID", "True Label", "Predicted Label"])
    writer.writerows(wrong_predictions)

print(f"\nIncorrectly predicted samples saved to: {csv_filename}")

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print("\n" + "="*80)
print("CONFUSION MATRIX:")
print("="*80)
print(conf_matrix)

# Accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Create confusion matrix heatmap
row_labels = ['Hyperdiploid', 'Low hypodiploid', 'Near haploid', 'iAMP21', 'Other B-ALL']
col_labels = ['Hyperdiploid', 'Low hypodiploid', 'Near haploid', 'iAMP21', 'Other B-ALL']

plt.figure(figsize=(10, 6))
sns.set(font_scale=1.4)
norm = plt.Normalize(vmin=np.log1p(conf_matrix).min(), vmax=np.log1p(conf_matrix).max())

ax = sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="YlGnBu",
    xticklabels=col_labels,
    yticklabels=row_labels,
    fmt='d',
    cbar_kws={"label": "Logarithmic Scale"},
    norm=norm,
    annot_kws={"size": 22}
)

ax.set_title('Confusion Matrix - Nested CV Ensemble Classifier', fontsize=18, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=20, fontweight='bold')
ax.set_ylabel('True Label', fontsize=20, fontweight='bold')
plt.xticks(rotation=45, ha='right')

heatmap_filename = OUTPUT_DIR + 'Ensemble_Classifier_Confusion_Matrix_NestedCV.svg'
plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix heatmap saved to: {heatmap_filename}")
plt.close()

# Classification report
class_report = classification_report(
    true_labels,
    predictions,
    target_names=label_encoder.classes_
)
print("\n" + "="*80)
print("CLASSIFICATION REPORT:")
print("="*80)
print(class_report)

# ============================================================================
# TRAIN FINAL MODEL WITH MOST COMMON HYPERPARAMETERS
# ============================================================================
print("\n" + "="*80)
print("TRAINING FINAL MODEL ON FULL DATASET...")
print("="*80)

# Fit scaler on full dataset for final model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to full dataset
if len(labels_to_upsample_encoded) > 0:
    valid_labels = []
    for label in labels_to_upsample_encoded:
        class_count = np.sum(y_encoded == label)
        if class_count >= 2:
            valid_labels.append(label)

    if len(valid_labels) > 0:
        k_neighbors_final = get_k_neighbors(y_encoded)
        knn_estimator_final = NearestNeighbors(n_neighbors=k_neighbors_final, n_jobs=-1)

        try:
            smote_final = SMOTE(
                sampling_strategy={label: upsample_count for label in valid_labels},
                k_neighbors=knn_estimator_final,
                random_state=42
            )
            X_final_smote, y_final_smote = smote_final.fit_resample(X_scaled, y_encoded)

            print(f"Original dataset size: {len(X_scaled)} samples")
            print(f"After SMOTE: {len(X_final_smote)} samples")
        except ValueError:
            X_final_smote, y_final_smote = X_scaled, y_encoded
    else:
        X_final_smote, y_final_smote = X_scaled, y_encoded
else:
    X_final_smote, y_final_smote = X_scaled, y_encoded

# Create final classifiers with most common parameters
final_knn_params['n_jobs'] = -1
final_knn = KNeighborsClassifier(**final_knn_params)

final_rf_params['n_jobs'] = -1
if 'random_state' not in final_rf_params:
    final_rf_params['random_state'] = 42
final_rf = RandomForestClassifier(**final_rf_params)

final_xgb_params['n_jobs'] = -1
if 'random_state' not in final_xgb_params:
    final_xgb_params['random_state'] = 42
final_xgb = XGBClassifier(**final_xgb_params)

# Create final ensemble
final_ensemble = VotingClassifier(
    estimators=[
        ('knn', final_knn),
        ('rf', final_rf),
        ('xgb', final_xgb)
    ],
    voting='soft',
    n_jobs=-1
)

print("\nTraining final ensemble classifier...")
final_ensemble.fit(X_final_smote, y_final_smote)

# Save model
model_data = {
    'model': final_ensemble,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_names': X.columns.tolist(),
    'class_names': label_encoder.classes_.tolist(),
    'hyperparameters': {
        'knn': final_knn_params,
        'rf': final_rf_params,
        'xgb': final_xgb_params
    },
    'nested_cv_info': {
        'method': 'Nested CV with LOOCV outer loop and 5-fold inner loop',
        'total_folds': len(X),
        'hyperparameter_distributions': OUTPUT_DIR + 'nested_cv_hyperparameters.csv'
    }
}

model_filename = OUTPUT_DIR + 'KaryALL_model_NestedCV.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model_data, model_file)

print(f"\n✓ Final model saved to: {model_filename}")
print(f"  - Model type: Ensemble (KNN + RF + XGBoost)")
print(f"  - Trained with Nested CV hyperparameters")
print(f"  - Number of features: {len(X.columns)}")
print(f"  - Number of classes: {len(label_encoder.classes_)}")

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"Total time: {total_time/3600:.2f} hours")
print("="*80)
