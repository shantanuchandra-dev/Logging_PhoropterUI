import os
import csv
import glob

def compare_csvs(new_dir, backup_dir):
    print(f"Comparing {new_dir} vs {backup_dir}")
    new_files = glob.glob(os.path.join(new_dir, "*.csv"))
    
    total_changes = 0
    for nf in new_files:
        bn = os.path.basename(nf)
        bf = os.path.join(backup_dir, bn)
        if not os.path.exists(bf):
            print(f"[NEW VIDEO] {bn}")
            continue
            
        with open(nf, 'r') as f: new_rows = list(csv.DictReader(f))
        with open(bf, 'r') as f: old_rows = list(csv.DictReader(f))
        
        # Simple diff: compare row counts and specific JCC changes if possible
        if len(new_rows) != len(old_rows):
            print(f"[LENGTH CHANGE] {bn}: {len(old_rows)} -> {len(new_rows)} rows")
        
        # Check for state changes at same timestamps
        old_map = {r['Timestamp']: r for r in old_rows}
        changed_states = 0
        file_header_printed = False
        
        for nr in new_rows:
            ts = nr['Timestamp']
            if ts in old_map:
                if nr['Occluder_State'] != old_map[ts]['Occluder_State']:
                    if not file_header_printed:
                        print(f"\n--- Changes in {bn} ---")
                        print(f"{'Timestamp':<10} | {'Backup State':<25} | {'New State':<25}")
                        print("-" * 65)
                        file_header_printed = True
                    
                    print(f"{ts:<10} | {old_map[ts]['Occluder_State']:<25} | {nr['Occluder_State']:<25}")
                    changed_states += 1
        
        if changed_states > 0:
            total_changes += changed_states

    print(f"\nTotal Clinical State Updates across all videos: {total_changes}")

if __name__ == "__main__":
    compare_csvs("MatchedScreens", "MatchedScreens_backup_20260125_173741")
