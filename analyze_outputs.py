import os
import csv
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime

def clean_value(val):
    if pd.isna(val) or str(val).strip() == '':
        return None
    try:
        return float(val)
    except:
        return None

def process_dataframe(df):
    """
    Applies null-filling and anomaly correction.
    F1: X, F2: null, F3: X -> F2 = X
    """
    cols = ['R_SPH', 'R_CYL', 'R_AXIS', 'R_ADD', 'L_SPH', 'L_CYL', 'L_AXIS', 'L_ADD']
    corrections_count = 0
    
    # 1. Null-filling (up to 2 frames)
    for col in cols:
        if col not in df.columns: continue
        vals = [clean_value(v) for v in df[col]]
        
        for i in range(1, len(vals) - 1):
            if vals[i] is None:
                # Check for X null X
                if i+1 < len(vals) and vals[i-1] is not None and vals[i-1] == vals[i+1]:
                    vals[i] = vals[i-1]
                    corrections_count += 1
                # Check for X null null X
                elif i+2 < len(vals) and vals[i] is None and vals[i+1] is None and \
                     vals[i-1] is not None and vals[i-1] == vals[i+2]:
                    vals[i] = vals[i-1]
                    vals[i+1] = vals[i-1]
                    corrections_count += 2
        
        # 2. Anomaly Correction (OCR Digits Fix)
        # Scan for jumps > 1.0 from neighbors when neighbors are consistent
        for i in range(1, len(vals) - 1):
            p, c, n = vals[i-1], vals[i], vals[i+1]
            if p is not None and c is not None and n is not None:
                if abs(p - n) < 0.5 and abs(c - p) > 1.0 and abs(c - n) > 1.0:
                    vals[i] = p
                    corrections_count += 1
        
        df[col] = vals
        
    return df, corrections_count

def analyze_workflow(df):
    """
    Analyzes JCC and Monocular testing.
    """
    states = df['Occluder_State'].fillna('').tolist()
    
    # JCC Sequences
    seq_valid_count = 0
    jcc_types = ['Axis', 'Power']
    eyes = ['Right', 'Left']
    
    for eye in eyes:
        for jt in jcc_types:
            prefix = f"{eye}_{jt}"
            f1_found = False
            for s in states:
                if f"{prefix}_Flip1" in s: f1_found = True
                if f1_found and f"{prefix}_Flip2" in s:
                    seq_valid_count += 1
                    break
    
    # Monocular Tests
    # R Tested (Left Occluded), L Tested (Right Occluded)
    r_tested = any('Left_Occluded' in s for s in states)
    l_tested = any('Right_Occluded' in s for s in states)
    
    monocular_found = (1 if r_tested else 0) + (1 if l_tested else 0)
    
    unique_jcc = len([s for s in df['Occluder_State'].unique() if any(x in str(s) for x in ['Axis', 'Power'])])
    
    return {
        'JCC_Stages_Found': unique_jcc,
        'JCC_Sequences_Valid': seq_valid_count,
        'Monocular_Tests_Found': monocular_found
    }

def run_analysis():
    input_dir = "MatchedScreens"
    output_dir = "Analyzed_CSVs"
    master_file = "Sample/AI Optom Co-Pilot - Dataset + Trackr - Consolidated 800.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    summary_data = {}
    
    print(f"Processing {len(csv_files)} files...")
    for fpath in csv_files:
        eid = os.path.splitext(os.path.basename(fpath))[0]
        out_fpath = os.path.join(output_dir, f"{eid}.csv")
        
        # Incremental logic: Skip if already analyzed
        if os.path.exists(out_fpath):
            print(f"Skipping already analyzed: {eid}")
            # Still need to load it for master CSV update summary
            try:
                df_existing = pd.read_csv(out_fpath)
                workflow = analyze_workflow(df_existing)
                # ... repeat verdict logic ...
                # Actually, let's just re-analyze for summary mapping unless speed is critical, 
                # but we respect NOT re-writing or re-running expensive bits if we can.
            except: pass
            # For simplicity and correctness of master CSV, we re-run the logic but skip writing if user wants speed, 
            # though here "not analyze" typically means "don't process again".
            # I will skip the processing but keep it in summary_data for master CSV consistency.
            if eid not in summary_data:
                try:
                    df = pd.read_csv(out_fpath)
                    workflow = analyze_workflow(df)
                    status = 'POOR'
                    if workflow['JCC_Stages_Found'] >= 6 and workflow['JCC_Sequences_Valid'] >= 2 and workflow['Monocular_Tests_Found'] == 2:
                        status = 'COMPLETE'
                    elif workflow['JCC_Stages_Found'] >= 3 or workflow['JCC_Sequences_Valid'] >= 1:
                        status = 'PARTIAL'
                    summary_data[eid] = {
                        'Exam_Status': status,
                        'JCC_Stages': workflow['JCC_Stages_Found'],
                        'JCC_Sequences': workflow['JCC_Sequences_Valid'],
                        'Monocular_Tests': workflow['Monocular_Tests_Found'],
                        'Anomalies_Fixed': df['Anomalies_Fixed'].iloc[0] if 'Anomalies_Fixed' in df.columns else 0
                    }
                    continue
                except: continue

        try:
            df = pd.read_csv(fpath)
            if df.empty: continue
            
            # Step 1: Correction
            df, corrections = process_dataframe(df)
            
            # Step 2: Quality metadata
            cols = ['R_SPH', 'R_CYL', 'R_AXIS', 'R_ADD', 'L_SPH', 'L_CYL', 'L_AXIS', 'L_ADD']
            df['OCR_Fields_Read'] = df[cols].notna().sum(axis=1)
            df['Anomalies_Fixed'] = corrections 
            
            # Save cleaned CSV
            df.to_csv(out_fpath, index=False)
            
            # Step 3: Workflow analysis
            workflow = analyze_workflow(df)
            
            # Step 4: Final Verdict
            status = 'POOR'
            if workflow['JCC_Stages_Found'] >= 6 and workflow['JCC_Sequences_Valid'] >= 2 and workflow['Monocular_Tests_Found'] == 2:
                status = 'COMPLETE'
            elif workflow['JCC_Stages_Found'] >= 3 or workflow['JCC_Sequences_Valid'] >= 1:
                status = 'PARTIAL'
                
            summary_data[eid] = {
                'Exam_Status': status,
                'JCC_Stages': workflow['JCC_Stages_Found'],
                'JCC_Sequences': workflow['JCC_Sequences_Valid'],
                'Monocular_Tests': workflow['Monocular_Tests_Found'],
                'Anomalies_Fixed': corrections
            }
        except Exception as e:
            print(f"Error processing {eid}: {e}")

    # Step 5: Merge into Master
    if os.path.exists(master_file):
        print(f"Updating master CSV: {master_file}")
        master_df = pd.read_csv(master_file)
        
        for metric in ['Exam_Status', 'JCC_Stages', 'JCC_Sequences', 'Monocular_Tests', 'Anomalies_Fixed']:
            master_df[metric] = master_df['engagementId'].map(lambda x: summary_data.get(x, {}).get(metric, 'N/A'))
            
        master_df.to_csv(master_file, index=False)
        print("Master CSV updated successfully.")
    else:
        print(f"Warning: Master CSV {master_file} not found.")

if __name__ == "__main__":
    run_analysis()
