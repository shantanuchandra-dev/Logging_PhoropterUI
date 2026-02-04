"""
Output handling: Write results to CSV and generate reports.
"""
import csv
from pathlib import Path
from typing import List
from ..core.context import RowContext


def write_annotated_csv(rows: List[RowContext], output_path: Path):
    """
    Write annotated CSV with phase information.
    
    Args:
        rows: List of RowContext objects with phase_id and phase_name populated
        output_path: Path to output CSV file
    """
    if not rows:
        return
    
    # Build fieldnames
    base_fields = [
        "Timestamp", "R_SPH", "R_CYL", "R_AXIS", "R_ADD",
        "L_SPH", "L_CYL", "L_AXIS", "L_ADD", "PD",
        "Chart_Number", "Occluder_State", "Chart_Display",
        "OCR_Fields_Read", "Anomalies_Fixed",
    ]
    
    conversation_fields = [
        "Optometrist_Question", "Patient_Answer_Intent", "Patient_Confidence"
    ]
    
    phase_fields = ["Phase_ID", "Phase_Name"]
    
    fieldnames = base_fields + conversation_fields + phase_fields
    
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            row_dict = {
                "Timestamp": row.timestamp,
                "R_SPH": row.r_sph,
                "R_CYL": row.r_cyl,
                "R_AXIS": row.r_axis,
                "R_ADD": row.r_add,
                "L_SPH": row.l_sph,
                "L_CYL": row.l_cyl,
                "L_AXIS": row.l_axis,
                "L_ADD": row.l_add,
                "PD": row.pd,
                "Chart_Number": row.chart_number,
                "Occluder_State": row.occluder_state,
                "Chart_Display": row.chart_display,
                "OCR_Fields_Read": row.ocr_fields_read,
                "Anomalies_Fixed": row.anomalies_fixed,
                "Optometrist_Question": row.optometrist_question or "",
                "Patient_Answer_Intent": row.patient_answer_intent or "",
                "Patient_Confidence": row.patient_confidence or "",
                "Phase_ID": row.phase_id or "",
                "Phase_Name": row.phase_name or "",
            }
            writer.writerow(row_dict)


def generate_summary(rows: List[RowContext]) -> dict:
    """
    Generate summary statistics for a test session.
    
    Args:
        rows: List of RowContext objects
    
    Returns:
        Dict with summary statistics
    """
    if not rows:
        return {}
    
    # Count phases
    phase_counts = {}
    for row in rows:
        if row.phase_id:
            phase_counts[row.phase_id] = phase_counts.get(row.phase_id, 0) + 1
    
    # Extract final prescription
    final_rx = {}
    if rows:
        last_row = rows[-1]
        final_rx = {
            "right_eye": {
                "sph": last_row.r_sph,
                "cyl": last_row.r_cyl,
                "axis": last_row.r_axis,
                "add": last_row.r_add,
            },
            "left_eye": {
                "sph": last_row.l_sph,
                "cyl": last_row.l_cyl,
                "axis": last_row.l_axis,
                "add": last_row.l_add,
            },
        }
    
    # Calculate duration
    start_time = rows[0].timestamp if rows else ""
    end_time = rows[-1].timestamp if rows else ""
    
    return {
        "total_rows": len(rows),
        "phase_counts": phase_counts,
        "final_prescription": final_rx,
        "start_time": start_time,
        "end_time": end_time,
    }


def write_summary_report(summary: dict, output_path: Path):
    """
    Write summary report to text file.
    
    Args:
        summary: Summary dict from generate_summary()
        output_path: Path to output text file
    """
    with output_path.open('w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EYE TEST SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Rows: {summary['total_rows']}\n")
        f.write(f"Start Time: {summary['start_time']}\n")
        f.write(f"End Time: {summary['end_time']}\n\n")
        
        f.write("Phase Distribution:\n")
        f.write("-" * 40 + "\n")
        for phase, count in sorted(summary['phase_counts'].items()):
            f.write(f"  {phase}: {count} rows\n")
        f.write("\n")
        
        f.write("Final Prescription:\n")
        f.write("-" * 40 + "\n")
        rx = summary['final_prescription']
        f.write("Right Eye:\n")
        f.write(f"  SPH: {rx['right_eye']['sph']:+.2f}  CYL: {rx['right_eye']['cyl']:+.2f}  AXIS: {rx['right_eye']['axis']:.0f}°  ADD: {rx['right_eye']['add']:+.2f}\n")
        f.write("Left Eye:\n")
        f.write(f"  SPH: {rx['left_eye']['sph']:+.2f}  CYL: {rx['left_eye']['cyl']:+.2f}  AXIS: {rx['left_eye']['axis']:.0f}°  ADD: {rx['left_eye']['add']:+.2f}\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
