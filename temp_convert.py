#!/usr/bin/env python3
"""Update metadata paths from Windows to VM Linux format"""
import pandas as pd
import sys

# Read the CSV
df = pd.read_csv('ucec_grading_metadata.csv')

# Convert Windows paths to Linux VM paths
vm_base_path = '/mnt/data/shared-data/endometrial-cancer'

def convert_path(path_str):
    if pd.isna(path_str) or path_str == '':
        return ''
    # Extract patient_id/series_id/filename from Windows path
    parts = str(path_str).replace('\\', '/').split('/')
    # Find patient ID position (starts with C3L- or C3N-)
    for i, part in enumerate(parts):
        if part.startswith('C3L-') or part.startswith('C3N-'):
            relative = '/'.join(parts[i:])
            return f"{vm_base_path}/{relative}"
    return path_str

# Apply conversion
df['image_path'] = df['image_path'].apply(convert_path)
df['mask_path'] = df['mask_path'].apply(convert_path)

# Save
df.to_csv('ucec_grading_metadata.csv', index=False)

print("✅ Successfully updated ucec_grading_metadata.csv")
print(f"\n📊 Total entries: {len(df)}")
print(f"\n📂 Sample converted path:")
print(f"   {df['image_path'].iloc[0]}")
print(f"\n✅ All paths converted from Windows to VM format")
