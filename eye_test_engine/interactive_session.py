#!/usr/bin/env python3
"""
Interactive eye test session orchestrator.
Manages conversation flow with patient and phoropter API calls.
"""
import json
import subprocess
from typing import Optional, List, Dict
from pathlib import Path

from .core.state_machine import StateMachine
from .core.context import RowContext


class InteractiveSession:
    """Orchestrates an interactive eye test session."""
    
    def __init__(self, base_url: str = "https://rajasthan-royals.preprod.lenskart.com",
                 phoropter_id: str = "phoropter-1"):
        self.base_url = base_url
        self.phoropter_id = phoropter_id
        self.api_endpoint = f"{base_url}/phoropter/{phoropter_id}/run-tests"
        
        # Initialize state machine
        self.state_machine = StateMachine()
        
        # Phase name mapping
        self.phase_names = {
            "distance_vision": "Phase A: Distance Vision (Step 2.1)",
            "right_eye_refraction": "Phase B: Right Eye Refraction (Step 6.1)",
            "jcc_axis_right": "Phase E: JCC Axis Right (Step 6.2)",
            "jcc_power_right": "Phase F: JCC Power Right (Step 6.2)",
            "duochrome_right": "Phase G: Duochrome Right (Step 6.2)",
            "left_eye_refraction": "Phase D: Left Eye Refraction (Step 6.3)",
            "jcc_axis_left": "Phase H: JCC Axis Left (Step 6.4)",
            "jcc_power_left": "Phase I: JCC Power Left (Step 6.4)",
            "duochrome_left": "Phase J: Duochrome Left (Step 6.4)",
            "binocular_balance": "Phase K: Binocular Balance (Step 6.5)",
        }
        
        # Current test state
        self.current_phase = "distance_vision"
        self.current_row = self._init_row()
        self.session_history: List[RowContext] = []
        
        # Refraction state tracking
        self.snellen_charts = [
            "snellen_chart_200_150",
            "snellen_chart_100_80",
            "snellen_chart_70_60_50",
            "snellen_chart_40_30_25",
            "snellen_chart_25_20_15",
            "snellen_chart_20_20_20",
            "snellen_chart_20_15_10",
        ]
        self.current_chart_index = 0
        self.unable_read_count = 0
        self.jcc_flip_state = "flip1"  # flip1 or flip2
        self.jcc_cycle_count = 0
        
        # Chart mappings
        self.chart_map = {
            "echart_400": "chart_9",
            "snellen_chart_200_150": "chart_10",
            "snellen_chart_100_80": "chart_11",
            "snellen_chart_70_60_50": "chart_12",
            "snellen_chart_40_30_25": "chart_13",
            "snellen_chart_20_15_10": "chart_14",
            "snellen_chart_20_20_20": "chart_15",
            "snellen_chart_25_20_15": "chart_16",
            "duochrome": "chart_17",
            "jcc_chart": "chart_19",
        }
    
    def _init_row(self) -> RowContext:
        """Initialize a row with restart state."""
        return RowContext(
            timestamp="00:00",
            r_sph=0.0, r_cyl=0.0, r_axis=180.0, r_add=0.0,
            l_sph=0.0, l_cyl=0.0, l_axis=180.0, l_add=0.0,
            pd="", chart_number=-1,
            occluder_state="BINO",
            chart_display="",
            ocr_fields_read=0,
            anomalies_fixed=0,
        )
    
    def reset_phoropter(self):
        """Reset phoropter to neutral state."""
        cmd = f'curl -X POST {self.base_url}/phoropter/{self.phoropter_id}/reset'
        subprocess.run(cmd, shell=True, capture_output=True)
        print("✓ Phoropter reset to 0/0/180")
    
    def set_chart(self, chart_name: str):
        """Display a chart on the phoropter."""
        chart_id = self.chart_map.get(chart_name)
        if not chart_id:
            print(f"Warning: Unknown chart {chart_name}")
            return
        
        payload = {
            "test_cases": [{
                "chart": {
                    "tab": "Chart1",
                    "chart_items": [chart_id]
                }
            }]
        }
        
        cmd = f"""curl -X POST {self.api_endpoint} \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload)}'"""
        
        subprocess.run(cmd, shell=True, capture_output=True)
        self.current_row.chart_display = chart_name
        print(f"✓ Displaying: {chart_name}")
    
    def set_power(self, r_sph: float = None, r_cyl: float = None, r_axis: float = None,
                  l_sph: float = None, l_cyl: float = None, l_axis: float = None,
                  occluder: str = None):
        """Set power and occluder on phoropter."""
        # Build payload
        right_eye = {}
        if r_sph is not None:
            right_eye["sph"] = r_sph
            self.current_row.r_sph = r_sph
        if r_cyl is not None:
            right_eye["cyl"] = r_cyl
            self.current_row.r_cyl = r_cyl
        if r_axis is not None:
            right_eye["axis"] = r_axis
            self.current_row.r_axis = r_axis
        
        left_eye = {}
        if l_sph is not None:
            left_eye["sph"] = l_sph
            self.current_row.l_sph = l_sph
        if l_cyl is not None:
            left_eye["cyl"] = l_cyl
            self.current_row.l_cyl = l_cyl
        if l_axis is not None:
            left_eye["axis"] = l_axis
            self.current_row.l_axis = l_axis
        
        payload = {"test_cases": [{}]}
        
        if right_eye:
            payload["test_cases"][0]["right_eye"] = right_eye
        if left_eye:
            payload["test_cases"][0]["left_eye"] = left_eye
        
        # Map occluder and set JCC eye mode for non-JCC phases
        jcc_eye_mode = None
        is_jcc_phase = self.current_phase in ["jcc_axis_right", "jcc_power_right", "jcc_axis_left", "jcc_power_left"]
        
        if occluder:
            if occluder == "Left_Occluded":
                payload["test_cases"][0]["aux_lens"] = "AuxLensL"
                jcc_eye_mode = "R"  # Use L when left is occluded
            elif occluder == "Right_Occluded":
                payload["test_cases"][0]["aux_lens"] = "AuxLensR"
                jcc_eye_mode = "L"  # Use R when right is occluded
            elif occluder == "BINO":
                payload["test_cases"][0]["aux_lens"] = "OFF"
                jcc_eye_mode = "BINO"
            self.current_row.occluder_state = occluder
        
        cmd = f"""curl -X POST {self.api_endpoint} \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload)}'"""
        
        subprocess.run(cmd, shell=True, capture_output=True)
        print(f"✓ Power set: R({r_sph}/{r_cyl}/{r_axis}) L({l_sph}/{l_cyl}/{l_axis}) Occ: {occluder}")
        
        # Set JCC eye mode for non-JCC phases only
        # JCC phases handle their own state when chart is displayed
        if jcc_eye_mode and not is_jcc_phase:
            self.jcc_control(jcc_eye_mode)
            print(f"✓ JCC eye mode set: {jcc_eye_mode}")
    
    def jcc_control(self, action: str):
        """Perform JCC action (handle, increase, decrease, etc.)."""
        payload = {"test_cases": [{"jcc": action}]}
        
        cmd = f"""curl -X POST {self.api_endpoint} \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload)}'"""
        
        subprocess.run(cmd, shell=True, capture_output=True)
        print(f"✓ JCC action: {action}")
    
    def get_question(self) -> str:
        """Get current question based on phase and state."""
        phase_config = self.state_machine.protocol["phases"].get(self.current_phase, {})
        questions = phase_config.get("questions", [])
        
        if isinstance(questions, dict):
            # JCC phase with flip1/flip2
            if self.current_row.is_flip1:
                return questions.get("flip1", "")
            elif self.current_row.is_flip2:
                return questions.get("flip2", "")
            return questions.get("flip1", "")
        elif isinstance(questions, list) and questions:
            return questions[0]
        
        return "Please describe what you see."
    
    def get_intents(self) -> List[str]:
        """Get available intents for current phase."""
        phase_config = self.state_machine.protocol["phases"].get(self.current_phase, {})
        intents = phase_config.get("intents", [])
        
        if isinstance(intents, dict):
            # JCC phase
            if self.current_row.is_flip1:
                flip1_intents = intents.get("flip1", [])
                # Return empty list for flip1 (no response needed, auto-flip)
                return flip1_intents if isinstance(flip1_intents, list) else []
            elif self.current_row.is_flip2:
                flip2_intents = intents.get("flip2", [])
                return flip2_intents if isinstance(flip2_intents, list) else [flip2_intents]
            return []
        elif isinstance(intents, list):
            return intents
        
        return ["Responds to instruction."]
    
    def start_distance_vision(self):
        """Start Phase A: Distance Vision."""
        self.current_phase = "distance_vision"
        
        print("\n" + "="*60)
        print(self.phase_names[self.current_phase].upper())
        print("="*60)
        
        # Note: Frontend already calls resetPhoropter() before starting session
        # self.reset_phoropter()
        self.set_chart("echart_400")
        self.current_row.occluder_state = "BINO"
        
        question = self.get_question()
        intents = self.get_intents()
        
        return {
            "phase": self.phase_names[self.current_phase],
            "question": question,
            "intents": intents,
            "chart": "echart_400",
            "occluder": "BINO",
            "power": {
                "right": {"sph": 0.0, "cyl": 0.0, "axis": 180.0},
                "left": {"sph": 0.0, "cyl": 0.0, "axis": 180.0},
            }
        }
    
    def process_response(self, intent: str) -> Dict:
        """Process patient response and return next question."""
        # Record current row
        self.current_row.patient_answer_intent = intent
        self.session_history.append(self.current_row)
        
        # Process based on current phase
        if self.current_phase == "distance_vision":
            return self._process_distance_vision(intent)
        elif self.current_phase == "right_eye_refraction":
            return self._process_right_eye_refraction(intent)
        elif self.current_phase == "jcc_axis_right":
            return self._process_jcc_axis_right(intent)
        elif self.current_phase == "jcc_power_right":
            return self._process_jcc_power_right(intent)
        elif self.current_phase == "duochrome_right":
            return self._process_duochrome_right(intent)
        elif self.current_phase == "left_eye_refraction":
            return self._process_left_eye_refraction(intent)
        elif self.current_phase == "jcc_axis_left":
            return self._process_jcc_axis_left(intent)
        elif self.current_phase == "jcc_power_left":
            return self._process_jcc_power_left(intent)
        elif self.current_phase == "duochrome_left":
            return self._process_duochrome_left(intent)
        elif self.current_phase == "binocular_balance":
            return self._process_binocular_balance(intent)
        
        # Default: complete
        return {
            "phase": "complete",
            "status": "complete",
            "question": "Test complete!",
            "intents": [],
        }
    
    def _process_distance_vision(self, intent: str) -> Dict:
        """Process distance vision phase."""
        # Move to right eye refraction
        self.current_phase = "right_eye_refraction"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.current_chart_index = 0  # Start with largest chart
        self.unable_read_count = 0
        
        # Create new row
        self.current_row = self._init_row()
        self.current_row.occluder_state = "Left_Occluded"
        self.current_row.chart_display = self.snellen_charts[0]
        
        # Set phoropter
        self.set_chart(self.snellen_charts[0])
        self.set_power(occluder="Left_Occluded")
        
        return self._build_response()
    
    def _process_right_eye_refraction(self, intent: str) -> Dict:
        """Process right eye refraction with chart progression."""
        current_chart = self.snellen_charts[self.current_chart_index]
        
        if intent == "Able to read":
            # Move to next smaller chart
            if current_chart == "snellen_chart_20_20_20":
                # Target reached, move to JCC
                return self._transition_to_jcc_axis_right()
            elif self.current_chart_index < len(self.snellen_charts) - 1:
                self.current_chart_index += 1
                self.unable_read_count = 0
                self.current_row = self._copy_row_state()
                self.current_row.chart_display = self.snellen_charts[self.current_chart_index]
                self.set_chart(self.snellen_charts[self.current_chart_index])
            else:
                # Reached smallest chart, move to JCC
                return self._transition_to_jcc_axis_right()
        
        elif intent == "Blurry":
            # Add -0.25D SPH, stay on same chart
            self.current_row = self._copy_row_state()
            self.current_row.r_sph -= 0.25
            self.set_power(r_sph=self.current_row.r_sph, occluder="Left_Occluded")
            self.unable_read_count = 0
        
        elif intent == "Unable to read":
            # Add -0.25D SPH, stay on same chart
            self.current_row = self._copy_row_state()
            self.current_row.r_sph -= 0.25
            self.set_power(r_sph=self.current_row.r_sph, occluder="Left_Occluded")
            self.unable_read_count += 1
            
            # Check exit condition: 2 consecutive "Unable to read"
            if self.unable_read_count >= 2:
                return self._transition_to_jcc_axis_right()
        
        elif intent == "Getting better":
            # Continue with current power, move to smaller chart
            if self.current_chart_index < len(self.snellen_charts) - 1:
                self.current_chart_index += 1
                self.unable_read_count = 0
                self.current_row = self._copy_row_state()
                self.current_row.chart_display = self.snellen_charts[self.current_chart_index]
                self.set_chart(self.snellen_charts[self.current_chart_index])
            else:
                return self._transition_to_jcc_axis_right()
        
        return self._build_response()
    
    def _process_left_eye_refraction(self, intent: str) -> Dict:
        """Process left eye refraction (same logic as right eye)."""
        current_chart = self.snellen_charts[self.current_chart_index]
        
        if intent == "Able to read":
            if current_chart == "snellen_chart_20_20_20":
                return self._transition_to_jcc_axis_left()
            elif self.current_chart_index < len(self.snellen_charts) - 1:
                self.current_chart_index += 1
                self.unable_read_count = 0
                self.current_row = self._copy_row_state()
                self.current_row.chart_display = self.snellen_charts[self.current_chart_index]
                self.set_chart(self.snellen_charts[self.current_chart_index])
            else:
                return self._transition_to_jcc_axis_left()
        
        elif intent == "Blurry":
            self.current_row = self._copy_row_state()
            self.current_row.l_sph -= 0.25
            self.set_power(l_sph=self.current_row.l_sph, occluder="Right_Occluded")
            self.unable_read_count = 0
        
        elif intent == "Unable to read":
            self.current_row = self._copy_row_state()
            self.current_row.l_sph -= 0.25
            self.set_power(l_sph=self.current_row.l_sph, occluder="Right_Occluded")
            self.unable_read_count += 1
            
            if self.unable_read_count >= 2:
                return self._transition_to_jcc_axis_left()
        
        elif intent == "Getting better":
            if self.current_chart_index < len(self.snellen_charts) - 1:
                self.current_chart_index += 1
                self.unable_read_count = 0
                self.current_row = self._copy_row_state()
                self.current_row.chart_display = self.snellen_charts[self.current_chart_index]
                self.set_chart(self.snellen_charts[self.current_chart_index])
            else:
                return self._transition_to_jcc_axis_left()
        
        return self._build_response()
    
    def _process_jcc_axis_right(self, intent: str) -> Dict:
        """Process JCC axis refinement for right eye."""
        if self.jcc_flip_state == "flip1":
            # Auto-progress from Flip1 to Flip2
            if intent == "AUTO_FLIP":
                # This is the automatic call after 2 seconds
                # Call handle to flip from position 1 to position 2
                self.jcc_flip_state = "flip2"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Right_Axis_Flip2")
                self.jcc_control("handle")  # Flip to position 2
                return self._build_response()
            else:
                # Initial entry - show Flip1 and request auto-flip
                response = self._build_response()
                response['auto_flip'] = True  # Tell frontend to auto-progress
                response['flip_wait_seconds'] = 2
                return response
        
        elif self.jcc_flip_state == "flip2":
            # Process Flip2 response
            if "GAP Axis" in intent or "Flip 1" in intent:
                # Patient chose Flip 1 - Use JCC increase operation
                self.jcc_control("increase")  # Phoropter increases axis by 5°
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.r_axis += 5
                if self.current_row.r_axis > 180:
                    self.current_row.r_axis -= 180
                
                # Reset to Flip1 for next cycle
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self._update_state(occluder="Right_Axis_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            
            elif "RAM Axis" in intent or "Flip 2" in intent:
                # Patient chose Flip 2 - Use JCC decrease operation
                self.jcc_control("decrease")  # Phoropter decreases axis by 5°
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.r_axis -= 5
                if self.current_row.r_axis < 0:
                    self.current_row.r_axis += 180
                
                # Reset to Flip1 for next cycle
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self._update_state(occluder="Right_Axis_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            
            elif "Both Same" in intent or "Reverse" in intent:
                # Move to JCC Power
                return self._transition_to_jcc_power_right()
            
            elif "Repeat" in intent:
                # Repeat the flip cycle - reset to flip1 and request auto-flip
                self.jcc_flip_state = "flip1"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Right_Axis_Flip1")
                # Note: JCC handle was already called, just need to show Flip1 again
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
        
        return self._build_response()
    
    def _process_jcc_axis_left(self, intent: str) -> Dict:
        """Process JCC axis refinement for left eye."""
        if self.jcc_flip_state == "flip1":
            # Auto-progress from Flip1 to Flip2
            if intent == "AUTO_FLIP":
                self.jcc_flip_state = "flip2"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Left_Axis_Flip2")
                self.jcc_control("handle")
                return self._build_response()
            else:
                # Initial entry - show Flip1 and request auto-flip
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
        
        elif self.jcc_flip_state == "flip2":
            if "GAP Axis" in intent or "Flip 1" in intent:
                # Patient chose Flip 1 - Use JCC increase operation
                self.jcc_control("increase")  # Phoropter increases axis by 5°
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.l_axis += 5
                if self.current_row.l_axis > 180:
                    self.current_row.l_axis -= 180
                
                # Reset to Flip1 for next cycle
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self._update_state(occluder="Left_Axis_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            
            elif "RAM Axis" in intent or "Flip 2" in intent:
                # Patient chose Flip 2 - Use JCC decrease operation
                self.jcc_control("decrease")  # Phoropter decreases axis by 5°
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.l_axis -= 5
                if self.current_row.l_axis < 0:
                    self.current_row.l_axis += 180
                
                # Reset to Flip1 for next cycle
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self._update_state(occluder="Left_Axis_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            
            elif "Both Same" in intent or "Reverse" in intent:
                return self._transition_to_jcc_power_left()
            
            elif "Repeat" in intent:
                self.jcc_flip_state = "flip1"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Left_Axis_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
        
        return self._build_response()
    
    def _process_jcc_power_right(self, intent: str) -> Dict:
        """Process JCC power refinement for right eye."""
        if self.jcc_flip_state == "flip1":
            # Auto-progress from Flip1 to Flip2
            if intent == "AUTO_FLIP":
                self.jcc_flip_state = "flip2"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Right_Power_Flip2")
                self.jcc_control("handle")
                return self._build_response()
            else:
                # Initial entry - show Flip1 and request auto-flip
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
        
        elif self.jcc_flip_state == "flip2":
            if "GAP Power" in intent or "Flip 1" in intent:
                # Patient chose Flip 1 - Use JCC increase operation
                self.jcc_control("increase")  # Phoropter increases cylinder by 0.25D
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.r_cyl += 0.25
                
                # Reset to Flip1 for next cycle
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self._update_state(occluder="Right_Power_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            
            elif "RAM Power" in intent or "Flip 2" in intent:
                # Patient chose Flip 2 - Use JCC decrease operation
                self.jcc_control("decrease")  # Phoropter decreases cylinder by 0.25D
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.r_cyl -= 0.25
                
                # Reset to Flip1 for next cycle
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self._update_state(occluder="Right_Power_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            
            elif "Both Same" in intent or "Reverse" in intent or (self.current_row.r_cyl == 0.0 and "GAP" in intent):
                return self._transition_to_duochrome_right()
            
            elif "Repeat" in intent:
                self.jcc_flip_state = "flip1"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Right_Power_Flip1")
                # Already at Flip1, just request auto-flip
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
        
        return self._build_response()
    
    def _process_jcc_power_left(self, intent: str) -> Dict:
        """Process JCC power refinement for left eye."""
        if self.jcc_flip_state == "flip1":
            # Auto-progress from Flip1 to Flip2
            if intent == "AUTO_FLIP":
                self.jcc_flip_state = "flip2"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Left_Power_Flip2")
                self.jcc_control("handle")
                return self._build_response()
            else:
                # Initial entry - show Flip1 and request auto-flip
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
        
        elif self.jcc_flip_state == "flip2":
            if "GAP Power" in intent or "Flip 1" in intent:
                # Patient chose Flip 1 - Use JCC increase operation
                self.jcc_control("increase")  # Phoropter increases cylinder by 0.25D
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.l_cyl += 0.25
                
                # Reset to Flip1 for next cycle
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self._update_state(occluder="Left_Power_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            
            elif "RAM Power" in intent or "Flip 2" in intent:
                # Patient chose Flip 2 - Use JCC decrease operation
                self.jcc_control("decrease")  # Phoropter decreases cylinder by 0.25D
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.l_cyl -= 0.25
                
                # Reset to Flip1 for next cycle
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self._update_state(occluder="Left_Power_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            
            elif "Both Same" in intent or "Reverse" in intent or (self.current_row.l_cyl == 0.0 and "GAP" in intent):
                return self._transition_to_duochrome_left()
            
            elif "Repeat" in intent:
                self.jcc_flip_state = "flip1"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Left_Power_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
        
        return self._build_response()
    
    def _process_duochrome_right(self, intent: str) -> Dict:
        """Process duochrome test for right eye."""
        # Adjust SPH based on response
        if intent == "Red":
            self.current_row = self._copy_row_state()
            self.current_row.r_sph += 0.25
            self.set_power(r_sph=self.current_row.r_sph, occluder="Left_Occluded")
        elif intent == "Green":
            self.current_row = self._copy_row_state()
            self.current_row.r_sph -= 0.25
            self.set_power(r_sph=self.current_row.r_sph, occluder="Left_Occluded")
        
        # Move to left eye refraction
        return self._transition_to_left_eye_refraction()
    
    def _process_duochrome_left(self, intent: str) -> Dict:
        """Process duochrome test for left eye."""
        if intent == "Red":
            self.current_row = self._copy_row_state()
            self.current_row.l_sph += 0.25
            self.set_power(l_sph=self.current_row.l_sph, occluder="Right_Occluded")
        elif intent == "Green":
            self.current_row = self._copy_row_state()
            self.current_row.l_sph -= 0.25
            self.set_power(l_sph=self.current_row.l_sph, occluder="Right_Occluded")
        
        # Move to binocular balance
        return self._transition_to_binocular_balance()
    
    def _process_binocular_balance(self, intent: str) -> Dict:
        """Process binocular balance phase."""
        # Test complete
        return {
            "phase": "complete",
            "status": "complete",
            "question": "Test complete!",
            "intents": [],
        }
    
    def _copy_row_state(self) -> RowContext:
        """Copy current row state to new row."""
        new_row = self._init_row()
        new_row.r_sph = self.current_row.r_sph
        new_row.r_cyl = self.current_row.r_cyl
        new_row.r_axis = self.current_row.r_axis
        new_row.l_sph = self.current_row.l_sph
        new_row.l_cyl = self.current_row.l_cyl
        new_row.l_axis = self.current_row.l_axis
        new_row.occluder_state = self.current_row.occluder_state
        new_row.chart_display = self.current_row.chart_display
        return new_row
    
    def _update_state(self, occluder: str = None, chart: str = None):
        """Update occluder and/or chart state and refresh derived fields."""
        if occluder is not None:
            self.current_row.occluder_state = occluder
        if chart is not None:
            self.current_row.chart_display = chart
        # Recalculate derived fields after manual changes
        self.current_row.update_derived_fields()
    
    def _build_response(self) -> Dict:
        """Build response with current state."""
        question = self.get_question()
        intents = self.get_intents()
        
        # Get formatted phase name with letter (A, B, etc.)
        phase_display = self.phase_names.get(self.current_phase, self.current_phase)
        
        return {
            "phase": phase_display,
            "question": question,
            "intents": intents,
            "chart": self.current_row.chart_display,
            "occluder": self.current_row.occluder_state,
            "power": {
                "right": {
                    "sph": self.current_row.r_sph,
                    "cyl": self.current_row.r_cyl,
                    "axis": self.current_row.r_axis,
                },
                "left": {
                    "sph": self.current_row.l_sph,
                    "cyl": self.current_row.l_cyl,
                    "axis": self.current_row.l_axis,
                }
            }
        }
    
    def _transition_to_jcc_axis_right(self) -> Dict:
        """Transition to JCC axis refinement for right eye."""
        self.current_phase = "jcc_axis_right"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.jcc_flip_state = "flip1"
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Right_Axis_Flip1"
        self.current_row.chart_display = "jcc_chart"
        
        # JCC chart defaults to Flip 1 of Axis when displayed
        self.set_chart("jcc_chart")
        # Note: No need to call jcc_flip("R") - chart defaults to correct state
        
        # Tell frontend to auto-flip after 2 seconds
        response = self._build_response()
        response['auto_flip'] = True
        response['flip_wait_seconds'] = 2
        return response
    
    def _transition_to_jcc_axis_left(self) -> Dict:
        """Transition to JCC axis refinement for left eye."""
        self.current_phase = "jcc_axis_left"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.jcc_flip_state = "flip1"
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Left_Axis_Flip1"
        self.current_row.chart_display = "jcc_chart"
        
        # JCC chart already displayed, no need to call any JCC APIs
        # Note: Chart maintains its state, defaults to Flip 1
        
        # Tell frontend to auto-flip after 2 seconds
        response = self._build_response()
        response['auto_flip'] = True
        response['flip_wait_seconds'] = 2
        return response
    
    def _transition_to_jcc_power_right(self) -> Dict:
        """Transition to JCC power refinement for right eye."""
        self.current_phase = "jcc_power_right"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.jcc_flip_state = "flip1"
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Right_Power_Flip1"
        self.current_row.chart_display = "jcc_chart"
        
        self.jcc_control("power_axis_switch")  # Switch to power mode - this resets to Flip 1
        
        # Tell frontend to auto-flip after 2 seconds
        response = self._build_response()
        response['auto_flip'] = True
        response['flip_wait_seconds'] = 2
        return response
    
    def _transition_to_jcc_power_left(self) -> Dict:
        """Transition to JCC power refinement for left eye."""
        self.current_phase = "jcc_power_left"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.jcc_flip_state = "flip1"
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Left_Power_Flip1"
        self.current_row.chart_display = "jcc_chart"
        
        self.jcc_control("power_axis_switch")  # Switch to power mode - this resets to Flip 1
        
        # Tell frontend to auto-flip after 2 seconds
        response = self._build_response()
        response['auto_flip'] = True
        response['flip_wait_seconds'] = 2
        return response
    
    def _transition_to_duochrome_right(self) -> Dict:
        """Transition to duochrome test for right eye."""
        self.current_phase = "duochrome_right"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Left_Occluded"
        self.current_row.chart_display = "duochrome"
        
        self.set_chart("duochrome")
        self.set_power(occluder="Left_Occluded")
        
        return self._build_response()
    
    def _transition_to_duochrome_left(self) -> Dict:
        """Transition to duochrome test for left eye."""
        self.current_phase = "duochrome_left"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Right_Occluded"
        self.current_row.chart_display = "duochrome"
        
        self.set_chart("duochrome")
        self.set_power(occluder="Right_Occluded")
        
        return self._build_response()
    
    def _transition_to_left_eye_refraction(self) -> Dict:
        """Transition to left eye refraction."""
        self.current_phase = "left_eye_refraction"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.current_chart_index = 0  # Start with largest chart
        self.unable_read_count = 0
        
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Right_Occluded"
        self.current_row.chart_display = self.snellen_charts[0]
        
        self.set_chart(self.snellen_charts[0])
        self.set_power(occluder="Right_Occluded")
        
        return self._build_response()
    
    def _transition_to_binocular_balance(self) -> Dict:
        """Transition to binocular balance."""
        self.current_phase = "binocular_balance"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "BINO"
        self.current_row.chart_display = "snellen_chart_20_20_20"
        
        self.set_chart("snellen_chart_20_20_20")
        self.set_power(occluder="BINO")
        
        return self._build_response()
    
    def _determine_next_phase(self, intent: str) -> str:
        """Determine next phase based on current phase and intent."""
        phase_flow = {
            "distance_vision": "right_eye_refraction",
            "right_eye_refraction": "jcc_axis_right",
            "jcc_axis_right": "jcc_power_right",
            "jcc_power_right": "duochrome_right",
            "duochrome_right": "left_eye_refraction",
            "left_eye_refraction": "jcc_axis_left",
            "jcc_axis_left": "jcc_power_left",
            "jcc_power_left": "duochrome_left",
            "duochrome_left": "binocular_balance",
            "binocular_balance": "complete",
        }
        
        return phase_flow.get(self.current_phase, "complete")
    
    def _setup_phase(self, phase: str):
        """Setup phoropter for the given phase."""
        # Create new row for this phase
        prev_row = self.current_row
        self.current_row = self._init_row()
        
        # Copy power from previous row
        self.current_row.r_sph = prev_row.r_sph
        self.current_row.r_cyl = prev_row.r_cyl
        self.current_row.r_axis = prev_row.r_axis
        self.current_row.l_sph = prev_row.l_sph
        self.current_row.l_cyl = prev_row.l_cyl
        self.current_row.l_axis = prev_row.l_axis
        
        if phase == "right_eye_refraction":
            self.set_chart("snellen_chart_20_20_20")
            self.set_power(occluder="Left_Occluded")
            self.current_row.occluder_state = "Left_Occluded"
            self.current_row.chart_display = "snellen_chart_20_20_20"
            
        elif phase == "jcc_axis_right":
            self.set_chart("jcc_chart")
            # JCC chart defaults to Flip 1 of Axis - no API calls needed
            self.current_row.occluder_state = "Right_Axis_Flip1"
            self.current_row.chart_display = "jcc_chart"
            
        elif phase == "jcc_power_right":
            # Switch to power mode
            self.jcc_control("power_axis_switch")
            self.current_row.occluder_state = "Right_Power_Flip1"
            self.current_row.chart_display = "jcc_chart"
            
        elif phase == "duochrome_right":
            self.set_chart("duochrome")
            self.set_power(occluder="Left_Occluded")
            self.current_row.occluder_state = "Left_Occluded"
            self.current_row.chart_display = "duochrome"
            
        elif phase == "left_eye_refraction":
            self.set_chart("snellen_chart_20_20_20")
            self.set_power(occluder="Right_Occluded")
            self.current_row.occluder_state = "Right_Occluded"
            self.current_row.chart_display = "snellen_chart_20_20_20"
            
        elif phase == "jcc_axis_left":
            # JCC chart already displayed - no API calls needed
            self.current_row.occluder_state = "Left_Axis_Flip1"
            self.current_row.chart_display = "jcc_chart"
            
        elif phase == "jcc_power_left":
            # Switch to power mode
            self.jcc_control("power_axis_switch")
            self.current_row.occluder_state = "Left_Power_Flip1"
            self.current_row.chart_display = "jcc_chart"
            
        elif phase == "duochrome_left":
            self.set_chart("duochrome")
            self.set_power(occluder="Right_Occluded")
            self.current_row.occluder_state = "Right_Occluded"
            self.current_row.chart_display = "duochrome"
            
        elif phase == "binocular_balance":
            self.set_chart("snellen_chart_20_20_20")
            self.set_power(occluder="BINO")
            self.current_row.occluder_state = "BINO"
            self.current_row.chart_display = "snellen_chart_20_20_20"


def main():
    """Interactive CLI demo."""
    session = InteractiveSession()
    
    print("\n" + "="*60)
    print("EYE TEST ENGINE - INTERACTIVE SESSION")
    print("="*60)
    
    # Start test
    state = session.start_distance_vision()
    
    print(f"\n📋 Phase: {state['phase']}")
    print(f"👁️  Chart: {state['chart']}")
    print(f"🔒 Occluder: {state['occluder']}")
    print(f"\n❓ Question: {state['question']}")
    print(f"\n💬 Available Intents:")
    for i, intent in enumerate(state['intents'], 1):
        print(f"   {i}. {intent}")
    
    print(f"\n🔧 Power: R({state['power']['right']['sph']}/{state['power']['right']['cyl']}/{state['power']['right']['axis']})")
    print(f"          L({state['power']['left']['sph']}/{state['power']['left']['cyl']}/{state['power']['left']['axis']})")
    
    # Wait for user input
    print(f"\n{'='*60}")
    print("Select an intent number (1-{}) or 'q' to quit:".format(len(state['intents'])))
    choice = input("> ")
    
    if choice.lower() == 'q':
        print("Session ended.")
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(state['intents']):
            selected_intent = state['intents'][idx]
            print(f"\n✓ Patient response: {selected_intent}")
            
            # Process response
            next_state = session.process_response(selected_intent)
            
            print(f"\n📋 Next Phase: {next_state['phase']}")
            print(f"❓ Next Question: {next_state['question']}")
            print(f"\n💬 Available Intents:")
            for i, intent in enumerate(next_state['intents'], 1):
                print(f"   {i}. {intent}")
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")


if __name__ == "__main__":
    main()
