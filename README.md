# KaryALL: B-ALL Subtype Classifier

This repository contains the training code for the KaryALL classifier, a machine learning model for classifying B-cell acute lymphoblastic leukemia (B-ALL) aneuploid  subtypes based on genomic features.

## Overview

The classifier uses an ensemble approach combining three machine learning algorithms:
- **k-Nearest Neighbors (KNN)**
- **Random Forest (RF)**
- **XGBoost (XGB)**

The model is trained using Leave-One-Out Cross-Validation (LOOCV) and addresses class imbalance through SMOTE (Synthetic Minority Over-sampling Technique).

## Classified Subtypes

The classifier distinguishes between the following B-ALL subtypes:
1. Hyperdiploid
2. Low hypodiploid
3. Near haploid
4. iAMP21 / chr21 amplification
5. Other B-ALL

## Repository Contents

### Main Scripts

- **`KaryALL_training.py`**: Main training script for the ensemble classifier
  - Performs hyperparameter optimization (optional)
  - Trains ensemble classifier using LOOCV
  - Generates confusion matrix and performance metrics
  - Exports misclassified samples

- **`extract_iAMP21_features.py`**: Feature extraction for iAMP21-specific genomic positions
  - Extracts 24 discriminative genomic positions
  - Merges features with main dataset

### Configuration Files

- **`requirements.txt`**: Python package dependencies

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Feature Extraction (Optional)

Extract iAMP21-specific genomic features from chromosome 21 position data.

**Quick Test with Example Data:**

```bash
# The script is pre-configured to use example data
python extract_iAMP21_features.py
```

**Extracting from Full Dataset:**

1. Edit `extract_iAMP21_features.py` and change line 18:
   ```python
   USE_EXAMPLE_DATA = False  # Switch from True to False
   ```

2. Run the extraction:
   ```bash
   python extract_iAMP21_features.py
   ```

**Required input files:**
- `formatted_data_example.csv`: Example genomic position data (5 samples, for testing)
- `merged_data_example.csv`: Example main dataset (5 samples, for testing)
- `formatted_data.csv`: Full genomic position data (for actual extraction)
- `merged_data.csv`: Full main dataset (for actual extraction)

**Output:**
- `iamp21_specific_features.csv`: Extracted 24 iAMP21-specific features
- `merged_data_with_formatted.csv`: Merged dataset (main features + iAMP21 features)

### 2. Classifier Training

**Quick Test with Example Data:**

To test the pipeline with the example dataset (10 samples):

```bash
# The script is pre-configured to use training_input.csv
# Simply run:
python KaryALL_training.py
```

**Training with Full Dataset:**

To train with your complete dataset:

1. Edit `KaryALL_training.py` and change line 20:
   ```python
   USE_EXAMPLE_DATA = False  # Switch from True to False
   ```

2. Run the training:
   ```bash
   python KaryALL_training.py
   ```

**Required input files:**
- `training_input.csv`: Example dataset (10 samples, for testing)
- `merged_data_with_formatted.csv`: Complete feature set with labels (for actual training)

**Output:**
- `Incorrectly_Predicted_Samples.csv`: List of misclassified samples
- `Ensemble_Classifier_Confusion_Matrix.svg`: Confusion matrix visualization
- `KaryALL_model.pkl`: Trained model file (for predictions on new data)
- Console output: Accuracy, classification report, and best hyperparameters

### Hyperparameter Optimization

The training script includes hyperparameter optimization using RandomizedSearchCV. This process can take several hours.

**To skip optimization and use pre-optimized parameters:**

1. Open `KaryALL_training.py`
2. Comment out the hyperparameter optimization section (lines ~40-140)
3. Uncomment the pre-optimized parameters section (lines ~145-165)

### 3. Using the Trained Model for Predictions

After training, you can use the saved model to classify new samples:

```python
import pickle
import pandas as pd

# Load the trained model
with open('KaryALL_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']

# Load new data
new_data = pd.read_csv('new_samples.csv')

# Ensure features are in the correct order
new_data = new_data[feature_names]

# Scale the data
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)

# Convert to class names
predicted_labels = label_encoder.inverse_transform(predictions)

# Display results
for i, (label, prob) in enumerate(zip(predicted_labels, probabilities)):
    max_prob = prob.max() * 100
    print(f"Sample {i+1}: {label} (confidence: {max_prob:.2f}%)")
```

## Data Preparation

The training data for KaryALL is derived from RNA-seq data processed through RNASeqCNV(https://www.nature.com/articles/s41375-022-01547-8) . This section describes how to prepare your data for training.

### Pipeline Overview

The data preparation involves two main steps:

1. **RNA-seq CNV Analysis** using a modified RNASeqCNV wrapper
2. **Feature Extraction** for iAMP21-specific genomic positions

### Processing RNA-seq Data

**Prerequisites:**
- Raw RNA-seq data (BAM files)
- Gene expression counts
- SNV information (VCF files or custom format)

**Pipeline:**

For processing RNA-seq data, we use a modified version of the RNASeqCNV wrapper from the [IntegrateALL pipeline](https://github.com/NadineWolgast/IntegrateALL). The modifications extract specific features required for machine learning classification:

- Chromosome-level CNV metrics (peak positions, peak maxima)
- Per-chromosome median weighted log2 fold changes
- Chromosome 21-specific features for iAMP21 detection

The modified RNASeqCNV wrapper processes each sample and generates:
- `{sampleID}_box_wdt.tsv`: Weighted boxplot statistics per chromosome
- `{sampleID}_smpSNPdata.csv`: SNV peak statistics
- `{sampleID}_formatted_data.csv`: Chr21 positional features
- `{sampleID}_ml_input.csv`: Combined feature set for classification

**Running the pipeline:**

```bash
# Example command for processing a sample
Rscript modified_RNASeqCNV_wrapper.R \
  --config config.txt \
  --metadata metadata.txt \
  --sample SAMPLE_ID \
  --outdir output/
```

For detailed information on the IntegrateALL pipeline, see: [IntegrateALL Repository](https://github.com/NadineWolgast/IntegrateALL)

### Extracting iAMP21 Features

After processing all samples with RNASeqCNV, extract iAMP21-specific features:

```bash
python extract_iAMP21_features.py
```

This extracts 24 discriminative genomic positions on chromosome 21 and merges them with the main CNV features.

### Merging Data for Training

Once all samples are processed:

1. Combine all `{sampleID}_ml_input.csv` files into a single dataset
2. Add sample metadata (sampleID, subtype labels)
3. The resulting file should match the format of `formated_data_example.csv`

**Example merging script:**

```python
import pandas as pd
import glob

# Read all ML input files
ml_files = glob.glob('output/*_ml_input.csv')
data_frames = []

for file in ml_files:
    df = pd.read_csv(file)
    sample_id = file.split('/')[-1].replace('_ml_input.csv', '')
    df['sampleID'] = sample_id
    data_frames.append(df)

# Combine all samples
combined_data = pd.concat(data_frames, ignore_index=True)

# Add subtype labels (from your annotation file)
annotations = pd.read_csv('sample_annotations.csv')
final_data = combined_data.merge(annotations, on='sampleID')

# Save for training
final_data.to_csv('merged_data_with_formatted.csv', index=False)
```

### Example Data

Example datasets are provided to demonstrate the expected input formats and enable testing:

**For Feature Extraction:**
- `formatted_data_example.csv`: 5 samples with chromosome 21 genomic positions
- `merged_data_example.csv`: 5 samples with CNV features

**For Classifier Training:**
- `training_input.csv`: 10 anonymized samples with complete feature set

Test the complete pipeline:

```bash
# Step 1: Test feature extraction
python extract_iAMP21_features.py

# Step 2: Test classifier training
python KaryALL_training.py
```

## Input Data Format

### Data File Structure

**Example Files (included in repository):**

| File | Samples | Features | Purpose |
|------|---------|----------|---------|
| `formatted_data_example.csv` | 5 | 119 | Chr21 position data for feature extraction |
| `merged_data_example.csv` | 5 | 163 | CNV features for merging |
| `training_input.csv` | 10 | 186 | Complete feature set for training |


### Training Data Structure

**Required columns:**
- `sampleID`: Unique sample identifier
- `subtype`: B-ALL subtype label (Hyperdiploid, Low hypodiploid, Near haploid, iAMP21, Other)
- Additional columns: Genomic features (numeric values)

Example structure:
```
sampleID,subtype,feature1,feature2,...,featureN
SAMPLE001,Hyperdiploid,0.523,1.234,...,0.891
SAMPLE002,iAMP21,0.125,0.456,...,1.023
...
```

## Model Performance

The classifier is evaluated using:
- **Leave-One-Out Cross-Validation (LOOCV)**: Ensures robust performance estimation
- **Confusion Matrix**: Shows classification performance per subtype
- **Classification Report**: Precision, recall, and F1-score for each class
- **Overall Accuracy**: Percentage of correctly classified samples

## Methodology

### Data Preprocessing
1. **Encoding**: Label encoding for target variable
2. **Normalization**: StandardScaler for feature scaling
3. **Upsampling**: SMOTE for minority classes (iAMP21, Near haploid)

### Training Strategy
- **Cross-Validation**: Leave-One-Out for maximum data utilization
- **Ensemble Method**: Soft voting across three classifiers
- **Class Imbalance**: SMOTE upsampling to majority class size
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold stratified CV

### Feature Selection
iAMP21-specific genomic positions were identified through feature importance analysis and represent chromosomal regions with high discriminative power for iAMP21 classification.

## Citation

If you use this classifier in your research, please cite:

```
https://doi.org/10.1101/2025.09.25.673987
```

## License

[Add your license information here]

## Contact

For questions or issues, please contact:
- [Nadine Wolgast; NadineWolgast@uksh.de]
- Or open an issue in this repository


