"""
iAMP21 Feature Extraction Script

This script extracts specific genomic positions that are particularly relevant
for iAMP21 classification. These positions were identified through feature
importance analysis and represent key chromosomal regions associated with
iAMP21 (intrachromosomal amplification of chromosome 21).

The extracted features are then merged with the main dataset to create
the final feature set used for training the classifier.
"""

import pandas as pd


# Load formatted data containing all genomic positions
# Switch between full dataset and example dataset
USE_EXAMPLE_DATA = True  # Set to False for full extraction

if USE_EXAMPLE_DATA:
    print("Loading EXAMPLE formatted data...")
    formatted_data = pd.read_csv('formatted_data_example.csv')
else:
    print("Loading formatted data...")
    formatted_data = pd.read_csv('formatted_data.csv')

# Fill missing values with 0
formatted_data.fillna(0, inplace=True)

# Define iAMP21-specific genomic positions
# These positions were identified through feature importance analysis
# and represent chromosomal regions with high discriminative power for iAMP21
iamp21_positions = [
    '36386148', '39428528', '37203112', '36990236', '39183488', '25772460',
    '33491716', '39349647', '37526358', '37101343', '37221736', '33643926',
    '37073170', '39321559', '38661780', '39236020', '32771792', '37267919',
    '32393012', '39314962', '36079389', '32279049', '33543491', '38747460'
]

# Extract sampleID and iAMP21-specific positions
print(f"Extracting {len(iamp21_positions)} iAMP21-specific genomic positions...")
iamp21_features = formatted_data[['sampleID'] + iamp21_positions].copy()

print(f"Extracted features shape: {iamp21_features.shape}")
print(f"Number of samples: {len(iamp21_features)}")
print(f"Number of iAMP21-specific features: {len(iamp21_positions)}")

# Display first few rows
print("\nFirst 5 rows of extracted iAMP21 features:")
print(iamp21_features.head())

# Save extracted features to CSV
output_filename = 'iamp21_specific_features.csv'
iamp21_features.to_csv(output_filename, index=False)
print(f"\niAMP21-specific features saved to: {output_filename}")

# Optional: Merge with main dataset if it exists
try:
    print("\nAttempting to merge with main dataset...")
    if USE_EXAMPLE_DATA:
        main_data = pd.read_csv('merged_data_example.csv')
    else:
        main_data = pd.read_csv('merged_data.csv')

    # Perform inner join based on sampleID
    merged_data = pd.merge(main_data, iamp21_features, on='sampleID', how='inner')

    # Save merged dataset
    merged_output = 'merged_data_with_formatted.csv'
    merged_data.to_csv(merged_output, index=False)

    print(f"Merged dataset shape: {merged_data.shape}")
    print(f"Merged dataset saved to: {merged_output}")

    # Display column information
    print(f"\nTotal features in merged dataset: {len(merged_data.columns)}")
    print(f"  - Original features: {len(main_data.columns)}")
    print(f"  - iAMP21-specific features added: {len(iamp21_positions)}")

except FileNotFoundError:
    print("\nNote: 'merged_data.csv' not found. Skipping merge step.")
    print("Only iAMP21-specific features have been extracted.")

print("\n" + "="*80)
print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nExtracted iAMP21-specific genomic positions:")
for i, pos in enumerate(iamp21_positions, 1):
    print(f"  {i:2d}. Position {pos}")
