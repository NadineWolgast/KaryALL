from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import csv
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
# Switch between full dataset and example dataset
USE_EXAMPLE_DATA = True  # Set to False for full training
OUTPUT_DIR = '../KaryALL_Optimization_Results/'  # Output directory for results

if USE_EXAMPLE_DATA:
    data = pd.read_csv('training_input.csv')
    print("="*80)
    print("USING EXAMPLE DATASET (training_input.csv)")
    print("="*80)
else:
    data = pd.read_csv('../merged_data_with_formatted.csv')
    print("="*80)
    print("USING FULL DATASET (merged_data_with_formatted.csv)")
    print("="*80)

print(f"Dataset size: {len(data)} samples")
print(f"Number of features: {len(data.columns) - 2}")  # -2 for sampleID and subtype
print(f"\nClass distribution:")
print(data['subtype'].value_counts())
print("="*80)

# Define features and target
X = data.drop(["sampleID", "subtype"], axis=1)
y = data["subtype"]

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# NOTE: Scaling is moved inside CV loop to prevent data leakage
# Each fold will have its own scaler fitted only on training data

# Labels for upsampling (only upsample classes that exist in the dataset)
all_minority_classes = ['iAMP21', 'Near haploid', 'Low hypodiploid']
labels_to_upsample = [label for label in all_minority_classes if label in y.values]

if len(labels_to_upsample) > 0:
    labels_to_upsample_encoded = label_encoder.transform(labels_to_upsample)
    print(f"\nClasses to be upsampled: {labels_to_upsample}")
else:
    labels_to_upsample_encoded = []
    print("\nNo minority classes found for upsampling")

# Number of samples to reach for upsampling
upsample_count = np.max(np.bincount(y_encoded))
print(f"Target sample count after SMOTE: {upsample_count}\n")


# Function to dynamically determine k_neighbors for SMOTE
def get_k_neighbors(y_train_smote):
    min_class_count = min(Counter(y_train_smote).values())
    return min(5, min_class_count - 1) if min_class_count > 1 else 1


# ============================================================================
# HYPERPARAMETER OPTIMIZATION WITH NESTED CV (Optional - currently disabled)
# ============================================================================
# NOTE: Hyperparameter optimization is currently DISABLED to save time
#
# This section implements NESTED CROSS-VALIDATION to avoid data leakage:
#   - Outer loop: Subset of data for validation (e.g., 50 samples for faster testing)
#   - Inner loop: 5-fold Stratified CV for hyperparameter search
#   - Scaling and SMOTE applied correctly within each fold
#
# For full nested CV validation with LOOCV, see KaryALL_training_nested_cv.py
# (takes ~2 hours for 395 samples)
#
# The pre-optimized parameters below were validated using full nested LOOCV.
#
# Uncomment the section below to run hyperparameter optimization with nested CV
'''
print("Starting Nested Cross-Validation for hyperparameter optimization...")
print("Note: This uses a subset of data for demonstration. For full LOOCV nested CV,")
print("      use the separate KaryALL_training_nested_cv.py script.")

from sklearn.model_selection import StratifiedKFold, train_test_split

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

# Use a subset for demonstration (remove stratify for full dataset)
X_tune, _, y_tune, _ = train_test_split(X, y_encoded, train_size=50,
                                         stratify=y_encoded, random_state=42)

# Outer CV loop (demonstration with small subset)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Inner CV loop for hyperparameter search
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_best_params = {'knn': [], 'rf': [], 'xgb': []}

print(f"\nRunning nested CV on {len(X_tune)} samples...")

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_tune, y_tune)):
    print(f"\nOuter fold {fold + 1}/5")

    # Split data (unscaled)
    X_train_outer = X_tune.iloc[train_idx]
    y_train_outer = y_tune[train_idx]

    # Scale training data
    fold_scaler = StandardScaler()
    X_train_scaled = fold_scaler.fit_transform(X_train_outer)

    # Apply SMOTE
    if len(labels_to_upsample_encoded) > 0:
        present_labels = [label for label in labels_to_upsample_encoded if label in y_train_outer]
        if len(present_labels) > 0:
            k_neighbors = get_k_neighbors(y_train_outer)
            knn_estimator = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=-1)
            try:
                smote = SMOTE(
                    sampling_strategy={label: upsample_count for label in present_labels},
                    k_neighbors=knn_estimator,
                    random_state=42
                )
                X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_outer)
            except ValueError:
                X_train_smote, y_train_smote = X_train_scaled, y_train_outer
        else:
            X_train_smote, y_train_smote = X_train_scaled, y_train_outer
    else:
        X_train_smote, y_train_smote = X_train_scaled, y_train_outer

    # Inner CV: Hyperparameter optimization
    print("  Optimizing KNN...")
    knn_search = RandomizedSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        knn_param_grid,
        n_iter=10,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    knn_search.fit(X_train_smote, y_train_smote)
    all_best_params['knn'].append(knn_search.best_params_)

    print("  Optimizing Random Forest...")
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_param_grid,
        n_iter=15,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    rf_search.fit(X_train_smote, y_train_smote)
    all_best_params['rf'].append(rf_search.best_params_)

    print("  Optimizing XGBoost...")
    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=42, n_jobs=-1),
        xgb_param_grid,
        n_iter=15,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    xgb_search.fit(X_train_smote, y_train_smote)
    all_best_params['xgb'].append(xgb_search.best_params_)

# Get most common hyperparameters
def get_most_common_params(param_list):
    param_strings = [str(sorted(p.items())) for p in param_list]
    from collections import Counter
    counter = Counter(param_strings)
    most_common_str = counter.most_common(1)[0][0]
    for p in param_list:
        if str(sorted(p.items())) == most_common_str:
            return p
    return param_list[0]

best_knn_params = get_most_common_params(all_best_params['knn'])
best_rf_params = get_most_common_params(all_best_params['rf'])
best_xgb_params = get_most_common_params(all_best_params['xgb'])

print("\n" + "="*80)
print("BEST HYPERPARAMETERS FOUND:")
print("="*80)
print(f"\nKNN: {best_knn_params}")
print(f"\nRandom Forest: {best_rf_params}")
print(f"\nXGBoost: {best_xgb_params}")
print("\n" + "="*80)
'''

# ============================================================================
# USE PRE-OPTIMIZED HYPERPARAMETERS
# ============================================================================
print("Using pre-optimized hyperparameters...")

best_knn_params = {
    'metric': 'manhattan',
    'n_neighbors': 3,
    'weights': 'distance'
}

best_rf_params = {
    'bootstrap': False,
    'max_depth': 95,
    'max_features': 'sqrt',
    'min_samples_leaf': 5,
    'min_samples_split': 6,
    'n_estimators': 150
}

best_xgb_params = {
    'colsample_bytree': 0.3538836655943906,
    'learning_rate': 0.03495151419409696,
    'max_depth': 9,
    'n_estimators': 105,
    'subsample': 0.6453037482886064
}

print("\n" + "="*80)
print("HYPERPARAMETERS BEING USED:")
print("="*80)
print(f"\nKNN: {best_knn_params}")
print(f"\nRandom Forest: {best_rf_params}")
print(f"\nXGBoost: {best_xgb_params}")
print("\n" + "="*80)


# ============================================================================
# LEAVE-ONE-OUT CROSS-VALIDATION WITH ENSEMBLE CLASSIFIER
# ============================================================================
print("\n\nStarting Leave-One-Out Cross-Validation...")

# Initialize Leave-One-Out Cross-Validation
loo = LeaveOneOut()

# Lists to store predictions and true labels
predictions = []
true_labels = []
sample_ids = []

# Perform Leave-One-Out Cross-Validation
for i, (train_index, test_index) in enumerate(loo.split(X)):
    if (i + 1) % 10 == 0:
        print(f"Processing sample {i + 1}/{len(X)}...")

    # Split data (unscaled)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    sample_id = data.iloc[test_index[0]]["sampleID"]

    # Fit scaler on training data only (prevents data leakage)
    fold_scaler = StandardScaler()
    X_train_scaled = fold_scaler.fit_transform(X_train)
    X_test_scaled = fold_scaler.transform(X_test)

    # Apply SMOTE only if there are minority classes to upsample
    if len(labels_to_upsample_encoded) > 0:
        # Filter labels_to_upsample to only include classes present in training set with at least 2 samples
        present_labels = []
        for label in labels_to_upsample_encoded:
            if label in y_train:
                # Count samples of this class in training set
                class_count = np.sum(y_train == label)
                if class_count >= 2:  # SMOTE requires at least 2 samples
                    present_labels.append(label)

        if len(present_labels) > 0:
            # Dynamic adjustment of k_neighbors
            k_neighbors = get_k_neighbors(y_train)

            # Create K-Neighbors estimator
            knn_estimator = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=-1)

            # Up-sample training data with SMOTE (on scaled data)
            try:
                smote = SMOTE(
                    sampling_strategy={label: upsample_count for label in present_labels},
                    k_neighbors=knn_estimator,
                    random_state=42
                )
                X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            except ValueError:
                # SMOTE failed (e.g., not enough samples), skip upsampling for this fold
                X_train_smote, y_train_smote = X_train_scaled, y_train
        else:
            # No minority classes with enough samples in training set, skip SMOTE
            X_train_smote, y_train_smote = X_train_scaled, y_train
    else:
        # No upsampling needed
        X_train_smote, y_train_smote = X_train_scaled, y_train

    # Create fresh classifier instances for this fold (prevents any state carryover)
    knn_params = best_knn_params.copy()
    knn_params['n_jobs'] = -1
    knn_classifier = KNeighborsClassifier(**knn_params)

    rf_params = best_rf_params.copy()
    rf_params['n_jobs'] = -1
    if 'random_state' not in rf_params:
        rf_params['random_state'] = 42
    rf_classifier = RandomForestClassifier(**rf_params)

    xgb_params = best_xgb_params.copy()
    xgb_params['n_jobs'] = -1
    if 'random_state' not in xgb_params:
        xgb_params['random_state'] = 42
    xgb_classifier = XGBClassifier(**xgb_params)

    # Create ensemble classifier
    ensemble_classifier = VotingClassifier(
        estimators=[
            ('knn', knn_classifier),
            ('rf', rf_classifier),
            ('xgb', xgb_classifier)
        ],
        voting='soft',
        n_jobs=-1
    )

    # Train ensemble classifier
    ensemble_classifier.fit(X_train_smote, y_train_smote)

    # Predict for test sample (using scaled test data)
    ensemble_pred = ensemble_classifier.predict(X_test_scaled)
    predictions.append(ensemble_pred[0])
    true_labels.append(y_test[0])
    sample_ids.append(sample_id)

print("Leave-One-Out Cross-Validation completed!")

# ============================================================================
# EVALUATION AND RESULTS
# ============================================================================

# Find incorrectly predicted entries
wrong_predictions = [
    (sample_id, label_encoder.inverse_transform([true])[0], label_encoder.inverse_transform([pred])[0])
    for sample_id, true, pred in zip(sample_ids, true_labels, predictions)
    if true != pred
]

# Output incorrectly predicted samples
print("\n" + "="*80)
print("INCORRECTLY PREDICTED SAMPLES:")
print("="*80)
for sample_id, true_label, pred_label in wrong_predictions:
    print(f"Sample ID: {sample_id}, True Label: {true_label}, Predicted Label: {pred_label}")

# Save as CSV file
csv_filename = OUTPUT_DIR + "Incorrectly_Predicted_Samples.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Sample ID", "True Label", "Predicted Label"])
    writer.writerows(wrong_predictions)

print(f"\nIncorrectly predicted samples saved to: {csv_filename}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print("\n" + "="*80)
print("CONFUSION MATRIX:")
print("="*80)
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Create confusion matrix heatmap
row_labels = ['Hyperdiploid', 'Low hypodiploid', 'Near haploid', 'iAMP21', 'Other B-ALL']
col_labels = ['Hyperdiploid', 'Low hypodiploid', 'Near haploid', 'iAMP21', 'Other B-ALL']

plt.figure(figsize=(10, 6))
sns.set(font_scale=1.4)

# Use logarithmic scale for better contrast with large differences
norm = plt.Normalize(vmin=np.log1p(conf_matrix).min(), vmax=np.log1p(conf_matrix).max())

# Draw heatmap with logarithmic color scale
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

# Title and axis labels with adjusted font size
ax.set_title('Confusion Matrix - Ensemble Classifier', fontsize=18, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=20, fontweight='bold')
ax.set_ylabel('True Label', fontsize=20, fontweight='bold')

# Display x-axis labels at an angle
plt.xticks(rotation=45, ha='right')

# Save heatmap
heatmap_filename = OUTPUT_DIR + 'Ensemble_Classifier_Confusion_Matrix.svg'
plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix heatmap saved to: {heatmap_filename}")

# Close plot to free memory
plt.close()

# Generate classification report
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
# TRAIN FINAL MODEL ON FULL DATASET AND SAVE
# ============================================================================
print("\n" + "="*80)
print("TRAINING FINAL MODEL ON FULL DATASET...")
print("="*80)

# For the final production model, fit scaler on full dataset
# (this is correct - the final model needs consistent scaling for new data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to full dataset for final model (if minority classes exist with enough samples)
if len(labels_to_upsample_encoded) > 0:
    # Check which classes have enough samples for SMOTE
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
            print("\nClass distribution after SMOTE:")
            for class_label, count in sorted(Counter(y_final_smote).items()):
                class_name = label_encoder.inverse_transform([class_label])[0]
                print(f"  {class_name}: {count} samples")
        except ValueError:
            # SMOTE failed, use original data
            X_final_smote, y_final_smote = X_scaled, y_encoded
            print(f"Dataset size: {len(X_scaled)} samples (SMOTE failed, using original data)")
            print("\nClass distribution:")
            for class_label, count in sorted(Counter(y_encoded).items()):
                class_name = label_encoder.inverse_transform([class_label])[0]
                print(f"  {class_name}: {count} samples")
    else:
        # No classes with enough samples for SMOTE
        X_final_smote, y_final_smote = X_scaled, y_encoded
        print(f"Dataset size: {len(X_scaled)} samples (not enough samples for SMOTE)")
        print("\nClass distribution:")
        for class_label, count in sorted(Counter(y_encoded).items()):
            class_name = label_encoder.inverse_transform([class_label])[0]
            print(f"  {class_name}: {count} samples")
else:
    # No upsampling needed for final model
    X_final_smote, y_final_smote = X_scaled, y_encoded

    print(f"Dataset size: {len(X_scaled)} samples (no SMOTE applied)")
    print("\nClass distribution:")
    for class_label, count in sorted(Counter(y_encoded).items()):
        class_name = label_encoder.inverse_transform([class_label])[0]
        print(f"  {class_name}: {count} samples")

# Create final ensemble classifier with best parameters
final_ensemble = VotingClassifier(
    estimators=[
        ('knn', knn_classifier),
        ('rf', rf_classifier),
        ('xgb', xgb_classifier)
    ],
    voting='soft',
    n_jobs=-1
)

# Train final model
print("\nTraining final ensemble classifier...")
final_ensemble.fit(X_final_smote, y_final_smote)

# Package model with preprocessing components
model_data = {
    'model': final_ensemble,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_names': X.columns.tolist(),
    'class_names': label_encoder.classes_.tolist(),
    'hyperparameters': {
        'knn': best_knn_params,
        'rf': best_rf_params,
        'xgb': best_xgb_params
    }
}

# Save model as pickle file
model_filename = OUTPUT_DIR + 'KaryALL_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model_data, model_file)

print(f"\n✓ Final model saved to: {model_filename}")
print(f"  - Model type: Ensemble (KNN + RF + XGBoost)")
print(f"  - Number of features: {len(X.columns)}")
print(f"  - Number of classes: {len(label_encoder.classes_)}")
print(f"  - Includes: model, scaler, label_encoder, feature_names, hyperparameters")

# Display usage example
print("\n" + "-"*80)
print("USAGE EXAMPLE:")
print("-"*80)
print("""
# Load the model
import pickle
with open('KaryALL_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']

# Prepare new data (must have same features as training data)
import pandas as pd
new_data = pd.read_csv('new_samples.csv')
new_data = new_data[model_data['feature_names']]  # Ensure correct feature order

# Scale and predict
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)

# Convert to class names
predicted_labels = label_encoder.inverse_transform(predictions)
print(predicted_labels)
""")

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
