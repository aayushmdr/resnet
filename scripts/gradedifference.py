import pandas as pd
import tarfile
import os
import io

# Paths
CLINICAL_TAR = './data/clinical/clinical.project-cptac-3.2026-03-02.tar.gz'
NIFTI_DIR = './data/raw_nifti'

def create_grading_pipeline(tar_path, image_dir):
    # 1. Load Clinical Data
    with tarfile.open(tar_path, 'r:gz') as gz:
        # Search for the clinical.tsv inside the archive
        clinical_file = next((f for f in gz.getmembers() if 'clinical.tsv' in f.name), None)
        if not clinical_file:
            print("Error: clinical.tsv not found in tar.")
            return pd.DataFrame()
        
        f = gz.extractfile(clinical_file)
        df = pd.read_csv(io.BytesIO(f.read()), sep='\t')

    # 2. Extract specific columns for UCEC Grading
    # Filtering for the specific cancer type in CPTAC-UCEC
    ucec_mask = df['diagnoses.primary_diagnosis'] == 'Endometrioid adenocarcinoma, NOS'
    df_grade = df[ucec_mask][['cases.submitter_id', 'diagnoses.tumor_grade']].drop_duplicates()

    # 3. Clean Labels
    grade_map = {'G1': 0, 'G2': 1, 'G3': 2}
    df_grade = df_grade[df_grade['diagnoses.tumor_grade'].isin(grade_map.keys())]
    df_grade['label'] = df_grade['diagnoses.tumor_grade'].map(grade_map)

    final_data = []
    
    # 4. Map to Folder Structure (PatientID -> SeriesID -> Files)
    # We iterate through the directory structure directly instead of a blind os.walk
    for patient_id in os.listdir(image_dir):
        patient_path = os.path.join(image_dir, patient_id)
        if not os.path.isdir(patient_path): continue
        
        # Check if this patient exists in our clinical grade list
        patient_info = df_grade[df_grade['cases.submitter_id'] == patient_id]
        if patient_info.empty: continue
        
        grade_str = patient_info.iloc[0]['diagnoses.tumor_grade']
        label_int = patient_info.iloc[0]['label']

        for series_id in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series_id)
            if not os.path.isdir(series_path): continue
            
            files = os.listdir(series_path)
            
            # --- CRITICAL FIX: Separate Image from Mask ---
            img_file = next((f for f in files if f == "image.nii.gz"), None)
            mask_file = next((f for f in files if f.startswith("mask_UTERUS")), None)

            if img_file:
                final_data.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_path': os.path.abspath(os.path.join(series_path, img_file)),
                    'mask_path': os.path.abspath(os.path.join(series_path, mask_file)) if mask_file else None,
                    'grade': grade_str,
                    'label': label_int
                })

    return pd.DataFrame(final_data)

# Execute
df_classification = create_grading_pipeline(CLINICAL_TAR, NIFTI_DIR)

if not df_classification.empty:
    df_classification.to_csv('ucec_grading_metadata.csv', index=False)
    print(f"✅ Pipeline Ready! Total samples: {len(df_classification)}")
    print(df_classification['grade'].value_counts())
else:
    print("❌ No matches found. Check if patient IDs in clinical.tsv match folder names.")