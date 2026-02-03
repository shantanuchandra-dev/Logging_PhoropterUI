import os
import csv
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Single optometrist step thresholds
# Sphere/Cylinder/Add step (diopters)
SINGLE_STEP_D = 0.25
# Axis step (degrees)
SINGLE_STEP_AXIS = 5
# Numeric comparison epsilon
NUM_EQ_EPS = 1e-9

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
            # Read raw strings so we can preserve formatting (e.g. leading '+')
            df_raw = pd.read_csv(fpath, dtype=str)
            if df_raw.empty: continue

            # Prepare a DataFrame for numeric processing: convert known numeric cols using clean_value
            cols = ['R_SPH', 'R_CYL', 'R_AXIS', 'R_ADD', 'L_SPH', 'L_CYL', 'L_AXIS', 'L_ADD']
            for col in cols:
                if col in df_raw.columns:
                    df_raw[col] = df_raw[col].where(df_raw[col].notna(), '')

            df_for_process = df_raw.copy()
            for col in cols:
                if col in df_for_process.columns:
                    df_for_process[col] = [clean_value(v) for v in df_for_process[col]]

            # Step 1: Correction on numeric values
            df_processed, corrections = process_dataframe(df_for_process)

            # Step 1.5: Enforce single-optometrist step multiples
            # For SPH/CYL: if value is not a multiple of SINGLE_STEP_D, use previous value when available.
            # For AXIS: if value is not a multiple of SINGLE_STEP_AXIS, use previous value when available.
            sph_cyl_cols = [c for c in ['R_SPH', 'R_CYL', 'L_SPH', 'L_CYL'] if c in df_processed.columns]
            axis_cols = [c for c in ['R_AXIS', 'L_AXIS'] if c in df_processed.columns]

            # SPH/CYL enforcement
            for c in sph_cyl_cols:
                vals = df_processed[c].tolist()
                for i in range(len(vals)):
                    v = vals[i]
                    if pd.isna(v):
                        continue
                    # Check if v is an exact multiple of SINGLE_STEP_D
                    try:
                        q = round(float(v) / SINGLE_STEP_D)
                    except:
                        continue
                    if not np.isclose(float(v), q * SINGLE_STEP_D, atol=NUM_EQ_EPS, rtol=0):
                        # Use previous value if available
                        if i > 0 and not pd.isna(vals[i-1]):
                            vals[i] = vals[i-1]
                df_processed[c] = vals

            # AXIS enforcement
            for c in axis_cols:
                vals = df_processed[c].tolist()
                for i in range(len(vals)):
                    v = vals[i]
                    if pd.isna(v):
                        continue
                    try:
                        q = round(float(v) / SINGLE_STEP_AXIS)
                    except:
                        continue
                    if not np.isclose(float(v), q * SINGLE_STEP_AXIS, atol=NUM_EQ_EPS, rtol=0):
                        if i > 0 and not pd.isna(vals[i-1]):
                            vals[i] = vals[i-1]
                df_processed[c] = vals

            # Cylinder restrictions: cannot be less than 0. If violated, use previous value or set to 0.0
            cyl_cols = [c for c in ['R_CYL', 'L_CYL'] if c in df_processed.columns]
            for c in cyl_cols:
                vals = df_processed[c].tolist()
                for i in range(len(vals)):
                    v = vals[i]
                    if pd.isna(v):
                        continue
                    try:
                        fv = float(v)
                    except:
                        continue
                    if fv < -NUM_EQ_EPS:
                        if i > 0 and not pd.isna(vals[i-1]):
                            vals[i] = vals[i-1]
                        else:
                            vals[i] = 0.0
                df_processed[c] = vals

            # ADD power restrictions: cannot exceed +6.00. If violated, use previous value or cap to 6.0
            add_cols = [c for c in ['R_ADD', 'L_ADD'] if c in df_processed.columns]
            for c in add_cols:
                vals = df_processed[c].tolist()
                for i in range(len(vals)):
                    v = vals[i]
                    if pd.isna(v):
                        continue
                    try:
                        fv = float(v)
                    except:
                        continue
                    if fv > 6.0 + NUM_EQ_EPS:
                        if i > 0 and not pd.isna(vals[i-1]):
                            vals[i] = vals[i-1]
                        else:
                            vals[i] = 6.0
                df_processed[c] = vals

            # Step 2: Quality metadata
            df_processed['OCR_Fields_Read'] = df_processed[cols].notna().sum(axis=1)
            df_processed['Anomalies_Fixed'] = corrections

            # Restore string formatting for numeric columns, preserving leading '+' when original had it
            def format_cell(orig_str, val, col_name):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return ''
                # If original string represented the same numeric value, keep original formatting (preserves '+')
                try:
                    orig_num = clean_value(orig_str) if isinstance(orig_str, str) else None
                except:
                    orig_num = None
                if orig_str is not None and str(orig_str).strip() != '' and orig_num is not None and float(orig_num) == float(val):
                    return orig_str

                # Otherwise format the processed numeric value: prefer integer style for whole numbers
                try:
                    fv = float(val)
                except:
                    return ''
                # If this is a SPH/CYL column and value is exactly zero, show as 0.00 per request
                if col_name in ['R_SPH', 'R_CYL', 'L_SPH', 'L_CYL'] and abs(fv) < NUM_EQ_EPS:
                    s = '0.00'
                elif abs(fv - round(fv)) < 1e-9:
                    s = str(int(round(fv)))
                else:
                    s = ('{:.2f}'.format(fv)).rstrip('0').rstrip('.')
                if isinstance(orig_str, str) and orig_str.strip().startswith('+') and fv > 0:
                    s = '+' + s
                return s

            # Build final dataframe to save: start with df_raw for non-numeric columns
            final_df = df_raw.copy()
            for col in cols:
                if col in final_df.columns:
                    final_df[col] = [format_cell(orig, val, col) for orig, val in zip(df_raw[col].tolist(), df_processed[col].tolist())]

            # Step 2.5: De-duplicate phoropter states to remove repeated readings
            # Deduplicate on all right/left values + PD + Chart_Number + Occluder_State + Chart_Display
            dedupe_cols = cols + ['PD', 'Chart_Number', 'Occluder_State', 'Chart_Display']
            dedupe_cols = [c for c in dedupe_cols if c in final_df.columns]
            if dedupe_cols:
                # Build equality check that treats numeric fields by numeric equality
                num_cols = [c for c in cols if c in final_df.columns]

                # Numeric comparison using processed numeric values where available
                num_equal = pd.Series([True] * len(final_df))
                if 'df_processed' in locals():
                    for c in num_cols:
                        a = df_processed[c].astype(float) if c in df_processed.columns else pd.Series([np.nan]*len(final_df))
                        b = a.shift(1)
                        eq = (pd.isna(a) & pd.isna(b)) | np.isclose(a.fillna(np.nan), b.fillna(np.nan), atol=NUM_EQ_EPS, rtol=0)
                        num_equal &= eq
                else:
                    # fallback to string compare if processed numeric not available
                    cmp_num = final_df[num_cols].fillna('').astype(str)
                    num_equal = cmp_num.eq(cmp_num.shift(1)).all(axis=1)

                # Categorical comparison for the remaining key columns
                cat_cols = [c for c in ['PD', 'Chart_Number', 'Occluder_State', 'Chart_Display'] if c in final_df.columns]
                if cat_cols:
                    cmp_cat = final_df[cat_cols].fillna('').astype(str)
                    cat_equal = cmp_cat.eq(cmp_cat.shift(1)).all(axis=1)
                else:
                    cat_equal = pd.Series([True] * len(final_df))

                same_as_prev = num_equal & cat_equal
                final_df = final_df[~same_as_prev].reset_index(drop=True)

            # Save cleaned (and deduped) CSV
            final_df.to_csv(out_fpath, index=False)
            
            # Step 3: Workflow analysis (use the cleaned/saved dataframe)
            workflow = analyze_workflow(final_df)
            
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
