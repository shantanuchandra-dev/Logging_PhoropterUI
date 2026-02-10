#!/usr/bin/env python3
"""
Interactive eye test session orchestrator.
Manages conversation flow with patient and phoropter API calls.
"""
import json
import subprocess
from typing import Optional, List, Dict
from pathlib import Path

from core.state_machine import StateMachine
from core.context import RowContext


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
        
        # Previous state tracking for "Prev State" functionality
        self.previous_state = None
        self.show_prev_state_option = False
        
        # JCC Power at 0.0 tracking
        self.jcc_power_zero_flip1_count = 0
        
        # Spherical equivalent compensation tracking
        # Track if we're at a -0.50D threshold (e.g., -0.50, -1.00, -1.50, etc.)
        # When crossing into threshold: SPH +0.25D
        # When crossing out of threshold: SPH -0.25D
        self.r_at_cyl_threshold = False
        self.l_at_cyl_threshold = False
        
        # Refraction state tracking
        # All available charts for selection
        self.all_charts = [
            "echart_400",
            "snellen_chart_200_150",
            "snellen_chart_100_80",
            "snellen_chart_70_60_50",
            "snellen_chart_40_30_25",
            "snellen_chart_25_20_15",
            "snellen_chart_20_20_20",
            "snellen_chart_20_15_10",
        ]
        # Snellen charts only (for automatic progression during refraction)
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
        self.jcc_last_choice: Optional[str] = None
        self.jcc_same_choice_count = 0
        self.duochrome_last_choice: Optional[str] = None
        self.duochrome_same_choice_count = 0
        
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
            "bino_chart": "chart_20",
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

    def _reset_jcc_choice_tracking(self) -> None:
        """Reset JCC flip choice tracking for reversal detection."""
        self.jcc_last_choice = None
        self.jcc_same_choice_count = 0

    def _record_jcc_choice(self, choice: str) -> bool:
        """Record a JCC flip choice and return True on reversal after a streak.

        A reversal occurs when the patient selects the opposite flip after
        choosing the same flip at least once in a row.
        """
        if choice == self.jcc_last_choice:
            self.jcc_same_choice_count += 1
            return False

        reversal = self.jcc_last_choice is not None and self.jcc_same_choice_count >= 1
        self.jcc_last_choice = choice
        self.jcc_same_choice_count = 1
        return reversal

    def _reset_duochrome_choice_tracking(self) -> None:
        """Reset duochrome color choice tracking for reversal detection."""
        self.duochrome_last_choice = None
        self.duochrome_same_choice_count = 0

    def _record_duochrome_choice(self, choice: str) -> bool:
        """Record a duochrome color choice and return True on reversal after a streak.

        A reversal occurs when the patient selects the opposite color after
        choosing the same color at least once in a row.
        """
        if choice == self.duochrome_last_choice:
            self.duochrome_same_choice_count += 1
            return False

        reversal = (
            self.duochrome_last_choice is not None
            and self.duochrome_same_choice_count >= 1
        )
        self.duochrome_last_choice = choice
        self.duochrome_same_choice_count = 1
        return reversal
    
    def reset_phoropter(self):
        """Reset phoropter to neutral state."""
        cmd = f'curl -X POST {self.base_url}/phoropter/{self.phoropter_id}/reset'
        print(f"[CMD] {cmd}")
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
        print(f"[CMD] {cmd}")
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
            # Note: AuxLens control removed - phoropter handles this automatically
            # Only track occluder state and JCC eye mode mapping
            if occluder == "Left_Occluded":
                jcc_eye_mode = "R"  # Use L when left is occluded
            elif occluder == "Right_Occluded":
                jcc_eye_mode = "L"  # Use R when right is occluded
            elif occluder == "BINO":
                jcc_eye_mode = "BINO"
            self.current_row.occluder_state = occluder
        
        cmd = f"""curl -X POST {self.api_endpoint} \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload)}'"""
        print(f"[CMD] {cmd}")
        subprocess.run(cmd, shell=True, capture_output=True)
        print(f"✓ Power set: R({r_sph}/{r_cyl}/{r_axis}) L({l_sph}/{l_cyl}/{l_axis}) Occ: {occluder}")
        
        # Set JCC eye mode for non-JCC phases only
        # JCC phases handle their own state when chart is displayed
        if jcc_eye_mode and not is_jcc_phase:
            self.jcc_control(jcc_eye_mode)
            print(f"✓ JCC eye mode set: {jcc_eye_mode}")
    
    def set_power_with_prev_state(self, 
                                   prev_r_sph: float, prev_r_cyl: float, prev_r_axis: float,
                                   prev_l_sph: float, prev_l_cyl: float, prev_l_axis: float,
                                   r_sph: float, r_cyl: float, r_axis: float,
                                   l_sph: float, l_cyl: float, l_axis: float,
                                   prev_aux_lens: str = None, aux_lens: str = None):
        """Set power with previous state for accurate click calculations.
        
        This uses the vision correction API with previous state to ensure
        accurate click calculations when the agent's internal state might be out of sync.
        """
        # Build payload with previous and current state
        payload = {
            "test_cases": [{
                "case_id": 1,
                "prev_right_eye": {"sph": prev_r_sph, "cyl": prev_r_cyl, "axis": prev_r_axis},
                "prev_left_eye": {"sph": prev_l_sph, "cyl": prev_l_cyl, "axis": prev_l_axis},
                "right_eye": {"sph": r_sph, "cyl": r_cyl, "axis": r_axis},
                "left_eye": {"sph": l_sph, "cyl": l_cyl, "axis": l_axis}
            }]
        }
        
        # Add aux_lens if provided
        if prev_aux_lens:
            payload["test_cases"][0]["prev_aux_lens"] = prev_aux_lens
        if aux_lens:
            payload["test_cases"][0]["aux_lens"] = aux_lens
        
        # Update internal state
        self.current_row.r_sph = r_sph
        self.current_row.r_cyl = r_cyl
        self.current_row.r_axis = r_axis
        self.current_row.l_sph = l_sph
        self.current_row.l_cyl = l_cyl
        self.current_row.l_axis = l_axis
        
        cmd = f"""curl -X POST {self.api_endpoint} \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload)}'"""
        print(f"[CMD] {cmd}")
        subprocess.run(cmd, shell=True, capture_output=True)
        print(f"✓ Power set with prev state: R({r_sph}/{r_cyl}/{r_axis}) L({l_sph}/{l_cyl}/{l_axis})")
        print(f"  Previous state: R({prev_r_sph}/{prev_r_cyl}/{prev_r_axis}) L({prev_l_sph}/{prev_l_cyl}/{prev_l_axis})")
    
    def jcc_control(self, action: str):
        """Perform JCC action (handle, increase, decrease, etc.)."""
        payload = {"test_cases": [{"jcc": action}]}
        
        cmd = f"""curl -X POST {self.api_endpoint} \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload)}'"""
        print(f"[CMD] {cmd}")
        subprocess.run(cmd, shell=True, capture_output=True)
        print(f"✓ JCC action: {action}")
    
    def set_pinhole(self):
        """Set pinhole on the phoropter."""
        cmd = f"curl -X POST {self.base_url}/phoropter/phoropter-1/pinhole"
        print(f"[CMD] {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"✓ Pinhole activated")
        return result
    
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
            # Add "Prev State" option if we have a previous state to restore
            if self.show_prev_state_option and self.previous_state is not None:
                return intents + ["Prev State"]
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
        self.current_chart_index = 0
        self.set_chart(self.all_charts[0])
        self.current_row.occluder_state = "BINO"
        self.current_row.chart_display = self.all_charts[0]
        
        question = self.get_question()
        intents = self.get_intents()
        
        return {
            "phase": self.phase_names[self.current_phase],
            "question": question,
            "intents": intents,
            "chart": self.all_charts[0],
            "occluder": "BINO",
            "power": {
                "right": {"sph": 0.0, "cyl": 0.0, "axis": 180.0},
                "left": {"sph": 0.0, "cyl": 0.0, "axis": 180.0},
            },
            "chart_info": {
                "available_charts": self.all_charts,
                "current_index": self.current_chart_index,
                "current_chart": self.all_charts[self.current_chart_index]
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
    
    def switch_chart(self, chart_index: int) -> Dict:
        """Switch to a different chart during distance vision or refraction phase.
        
        Args:
            chart_index: Index of the chart in the appropriate chart list
            
        Returns:
            Response dict with updated state
        """
        # Determine which chart list to use based on current phase
        if self.current_phase == "distance_vision":
            chart_list = self.all_charts
            phase_name = "distance vision"
        elif self.current_phase in ["right_eye_refraction", "left_eye_refraction"]:
            chart_list = self.snellen_charts
            phase_name = "refraction"
        else:
            raise ValueError(f"Chart switching not allowed in phase: {self.current_phase}")
        
        # Validate chart index
        if chart_index < 0 or chart_index >= len(chart_list):
            raise ValueError(f"Invalid chart index: {chart_index}")
        
        # Update chart index
        self.current_chart_index = chart_index
        
        # Update current row
        self.current_row = self._copy_row_state()
        self.current_row.chart_display = chart_list[chart_index]
        
        # Set the chart on phoropter
        self.set_chart(chart_list[chart_index])
        
        print(f"✓ Switched to chart {chart_index}: {chart_list[chart_index]}")
        
        # Return updated state
        return self._build_response()
    
    def _process_distance_vision(self, intent: str) -> Dict:
        """Process distance vision phase."""
        if intent == "Unable to read":
            # Add pinhole and test again
            print(f"\n→ Patient unable to read, adding pinhole (keeping current chart: {self.current_row.chart_display})")
            self.set_pinhole()
            
            # Update question to ask with pinhole (keep current chart)
            self.current_row = self._copy_row_state()
            # Don't change chart_display - keep whatever chart is currently displayed
            
            # Return response with pinhole question
            response = self._build_response()
            response['question'] = "With pinhole: Can you see clearly now?"
            response['intents'] = ["Able to read with pinhole", "Still unable to read"]
            return response
        
        elif intent == "Able to read with pinhole":
            # Pinhole helped, move to right eye refraction
            print("✓ Pinhole improved vision, proceeding to refraction")
            return self._transition_to_right_eye_refraction()
        
        elif intent == "Still unable to read":
            # Pinhole didn't help, still move to refraction but flag for further evaluation
            print("⚠️ Pinhole did not improve vision, proceeding to refraction")
            return self._transition_to_right_eye_refraction()
        
        # Default: "Able to read" or "Blurry"
        return self._transition_to_right_eye_refraction()
    
    def _transition_to_right_eye_refraction(self) -> Dict:
        """Transition to right eye refraction."""
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
        self.set_power(occluder="Left_Occluded") #SHANTANUCHANDRA: Commented out to avoid SETTING POWER during transition
        
        return self._build_response()
    
    def _process_right_eye_refraction(self, intent: str) -> Dict:
        """Process right eye refraction with chart progression."""
        current_chart = self.snellen_charts[self.current_chart_index]
        
        # Handle "Prev State" intent to restore previous power
        if intent == "Prev State":
            if self.previous_state is not None:
                self.current_row = self._copy_row_from_dict(self.previous_state)
                self.set_power(r_sph=self.current_row.r_sph, occluder="Left_Occluded")
                self.previous_state = None
                self.show_prev_state_option = False
                print("✓ Restored to previous state")
            return self._build_response()
        
        # Reset prev state option for non-power-changing responses
        if intent not in ["Blurry", "Unable to read"]:
            self.show_prev_state_option = False
        
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
            # Save current state before making changes
            self.previous_state = {
                'r_sph': self.current_row.r_sph,
                'r_cyl': self.current_row.r_cyl,
                'r_axis': self.current_row.r_axis,
                'l_sph': self.current_row.l_sph,
                'l_cyl': self.current_row.l_cyl,
                'l_axis': self.current_row.l_axis,
                'occluder_state': self.current_row.occluder_state,
                'chart_display': self.current_row.chart_display,
            }
            # Add -0.25D SPH, stay on same chart
            self.current_row = self._copy_row_state()
            self.current_row.r_sph -= 0.25
            self.set_power(r_sph=self.current_row.r_sph, occluder="Left_Occluded")
            self.unable_read_count = 0
            # Enable "Prev State" option for next response
            self.show_prev_state_option = True
        
        elif intent == "Unable to read":
            # Save current state before making changes
            self.previous_state = {
                'r_sph': self.current_row.r_sph,
                'r_cyl': self.current_row.r_cyl,
                'r_axis': self.current_row.r_axis,
                'l_sph': self.current_row.l_sph,
                'l_cyl': self.current_row.l_cyl,
                'l_axis': self.current_row.l_axis,
                'occluder_state': self.current_row.occluder_state,
                'chart_display': self.current_row.chart_display,
            }
            # Add -0.25D SPH, stay on same chart
            self.current_row = self._copy_row_state()
            self.current_row.r_sph -= 0.25
            self.set_power(r_sph=self.current_row.r_sph, occluder="Left_Occluded")
            self.unable_read_count += 1
            # Enable "Prev State" option for next response
            self.show_prev_state_option = True
            
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
        
        # Handle "Prev State" intent to restore previous power
        if intent == "Prev State":
            if self.previous_state is not None:
                self.current_row = self._copy_row_from_dict(self.previous_state)
                self.set_power(l_sph=self.current_row.l_sph, occluder="Right_Occluded")
                self.previous_state = None
                self.show_prev_state_option = False
                print("✓ Restored to previous state")
            return self._build_response()
        
        # Reset prev state option for non-power-changing responses
        if intent not in ["Blurry", "Unable to read"]:
            self.show_prev_state_option = False
        
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
            # Save current state before making changes
            self.previous_state = {
                'r_sph': self.current_row.r_sph,
                'r_cyl': self.current_row.r_cyl,
                'r_axis': self.current_row.r_axis,
                'l_sph': self.current_row.l_sph,
                'l_cyl': self.current_row.l_cyl,
                'l_axis': self.current_row.l_axis,
                'occluder_state': self.current_row.occluder_state,
                'chart_display': self.current_row.chart_display,
            }
            self.current_row = self._copy_row_state()
            self.current_row.l_sph -= 0.25
            self.set_power(l_sph=self.current_row.l_sph, occluder="Right_Occluded")
            self.unable_read_count = 0
            # Enable "Prev State" option for next response
            self.show_prev_state_option = True
        
        elif intent == "Unable to read":
            # Save current state before making changes
            self.previous_state = {
                'r_sph': self.current_row.r_sph,
                'r_cyl': self.current_row.r_cyl,
                'r_axis': self.current_row.r_axis,
                'l_sph': self.current_row.l_sph,
                'l_cyl': self.current_row.l_cyl,
                'l_axis': self.current_row.l_axis,
                'occluder_state': self.current_row.occluder_state,
                'chart_display': self.current_row.chart_display,
            }
            self.current_row = self._copy_row_state()
            self.current_row.l_sph -= 0.25
            self.set_power(l_sph=self.current_row.l_sph, occluder="Right_Occluded")
            self.unable_read_count += 1
            # Enable "Prev State" option for next response
            self.show_prev_state_option = True
            
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
            if "Repeat" in intent:
                # Repeat the flip cycle - reset to flip1 and request auto-flip
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Right_Axis_Flip1")
                # Note: JCC handle was already called, just need to show Flip1 again
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            if "GAP Axis" in intent or "Flip 1" in intent:
                # Patient chose Flip 1 - Use JCC increase operation
                reversal = self._record_jcc_choice("flip1")
                self.jcc_control("increase")  # Phoropter increases axis by 5°
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.r_axis += 5
                if self.current_row.r_axis > 180:
                    self.current_row.r_axis -= 180

                if reversal:
                    return self._transition_to_jcc_power_right()
                
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
                reversal = self._record_jcc_choice("flip2")
                self.jcc_control("decrease")  # Phoropter decreases axis by 5°
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.r_axis -= 5
                if self.current_row.r_axis < 0:
                    self.current_row.r_axis += 180

                if reversal:
                    return self._transition_to_jcc_power_right()
                
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
            if "Repeat" in intent:
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Left_Axis_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            if "GAP Axis" in intent or "Flip 1" in intent:
                # Patient chose Flip 1 - Use JCC increase operation
                reversal = self._record_jcc_choice("flip1")
                self.jcc_control("increase")  # Phoropter increases axis by 5°
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.l_axis += 5
                if self.current_row.l_axis > 180:
                    self.current_row.l_axis -= 180

                if reversal:
                    return self._transition_to_jcc_power_left()
                
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
                reversal = self._record_jcc_choice("flip2")
                self.jcc_control("decrease")  # Phoropter decreases axis by 5°
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.l_axis -= 5
                if self.current_row.l_axis < 0:
                    self.current_row.l_axis += 180

                if reversal:
                    return self._transition_to_jcc_power_left()
                
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
            if "Repeat" in intent:
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Right_Power_Flip1")
                # Already at Flip1, just request auto-flip
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            if "GAP Power" in intent or "Flip 1" in intent:
                # Patient chose Flip 1 - Use JCC increase operation
                
                # Special handling: If cylinder is 0.0, cannot increase (would go positive)
                if self.current_row.r_cyl == 0.0:
                    self.jcc_power_zero_flip1_count += 1
                    
                    if self.jcc_power_zero_flip1_count == 1:
                        # First time: Repeat the flip cycle
                        print("⚠️  Cylinder is 0.0, cannot increase. Repeating flip cycle...")
                        self.jcc_control("handle")
                        self.jcc_flip_state = "flip1"
                        self.current_row = self._copy_row_state()
                        self._update_state(occluder="Right_Power_Flip1")
                        response = self._build_response()
                        response['auto_flip'] = True
                        response['flip_wait_seconds'] = 2
                        return response
                    else:
                        # Second time: Move to next phase
                        print("⚠️  Cylinder is 0.0 and patient chose Flip 1 again. Moving to duochrome...")
                        self.jcc_power_zero_flip1_count = 0  # Reset counter
                        return self._transition_to_duochrome_right()
                
                # Normal case: cylinder is not 0.0
                reversal = self._record_jcc_choice("flip1")
                
                # Check if we're currently at a -0.50D threshold before increase
                was_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
                
                self.jcc_control("increase")  # Phoropter increases cylinder by 0.25D
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.r_cyl += 0.25
                
                # Check if we crossed out of a -0.50D threshold
                now_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
                
                if was_at_threshold and not now_at_threshold:
                    # Crossed out of threshold (e.g., -0.50 → -0.25)
                    # Revert spherical equivalent compensation: SPH -0.25D
                    self.current_row.r_sph -= 0.25
                    print(f"✓ Spherical equivalent reversion: SPH decreased by -0.25D (now {self.current_row.r_sph:.2f}D)")
                    # Note: Phoropter handles this automatically, we just track it

                if reversal:
                    return self._transition_to_duochrome_right()
                
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
                reversal = self._record_jcc_choice("flip2")
                
                # Check if we're currently at a -0.50D threshold before decrease
                was_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
                
                self.jcc_control("decrease")  # Phoropter decreases cylinder by 0.25D
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.r_cyl -= 0.25
                
                # Check if we crossed into a -0.50D threshold
                now_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
                
                if not was_at_threshold and now_at_threshold:
                    # Crossed into threshold (e.g., -0.25 → -0.50)
                    # Apply spherical equivalent compensation: SPH +0.25D
                    self.current_row.r_sph += 0.25
                    print(f"✓ Spherical equivalent compensation: SPH increased by +0.25D (now {self.current_row.r_sph:.2f}D)")
                    # Note: Phoropter handles this automatically, we just track it

                if reversal:
                    return self._transition_to_duochrome_right()
                
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
            if "Repeat" in intent:
                self.jcc_control("handle")
                self.jcc_flip_state = "flip1"
                self.current_row = self._copy_row_state()
                self._update_state(occluder="Left_Power_Flip1")
                response = self._build_response()
                response['auto_flip'] = True
                response['flip_wait_seconds'] = 2
                return response
            if "GAP Power" in intent or "Flip 1" in intent:
                # Patient chose Flip 1 - Use JCC increase operation
                
                # Special handling: If cylinder is 0.0, cannot increase (would go positive)
                if self.current_row.l_cyl == 0.0:
                    self.jcc_power_zero_flip1_count += 1
                    
                    if self.jcc_power_zero_flip1_count == 1:
                        # First time: Repeat the flip cycle
                        print("⚠️  Cylinder is 0.0, cannot increase. Repeating flip cycle...")
                        self.jcc_control("handle")
                        self.jcc_flip_state = "flip1"
                        self.current_row = self._copy_row_state()
                        self._update_state(occluder="Left_Power_Flip1")
                        response = self._build_response()
                        response['auto_flip'] = True
                        response['flip_wait_seconds'] = 2
                        return response
                    else:
                        # Second time: Move to next phase
                        print("⚠️  Cylinder is 0.0 and patient chose Flip 1 again. Moving to duochrome...")
                        self.jcc_power_zero_flip1_count = 0  # Reset counter
                        return self._transition_to_duochrome_left()
                
                # Normal case: cylinder is not 0.0
                reversal = self._record_jcc_choice("flip1")
                
                # Check if we're currently at a -0.50D threshold before increase
                was_at_threshold = self._is_at_cyl_threshold(self.current_row.l_cyl)
                
                self.jcc_control("increase")  # Phoropter increases cylinder by 0.25D
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.l_cyl += 0.25
                
                # Check if we crossed out of a -0.50D threshold
                now_at_threshold = self._is_at_cyl_threshold(self.current_row.l_cyl)
                
                if was_at_threshold and not now_at_threshold:
                    # Crossed out of threshold (e.g., -0.50 → -0.25)
                    # Revert spherical equivalent compensation: SPH -0.25D
                    self.current_row.l_sph -= 0.25
                    print(f"✓ Spherical equivalent reversion: SPH decreased by -0.25D (now {self.current_row.l_sph:.2f}D)")
                    # Note: Phoropter handles this automatically, we just track it

                if reversal:
                    return self._transition_to_duochrome_left()
                
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
                reversal = self._record_jcc_choice("flip2")
                
                # Check if we're currently at a -0.50D threshold before decrease
                was_at_threshold = self._is_at_cyl_threshold(self.current_row.l_cyl)
                
                self.jcc_control("decrease")  # Phoropter decreases cylinder by 0.25D
                
                # Update internal state (phoropter handles actual value)
                self.current_row = self._copy_row_state()
                self.current_row.l_cyl -= 0.25
                
                # Check if we crossed into a -0.50D threshold
                now_at_threshold = self._is_at_cyl_threshold(self.current_row.l_cyl)
                
                if not was_at_threshold and now_at_threshold:
                    # Crossed into threshold (e.g., -0.25 → -0.50)
                    # Apply spherical equivalent compensation: SPH +0.25D
                    self.current_row.l_sph += 0.25
                    print(f"✓ Spherical equivalent compensation: SPH increased by +0.25D (now {self.current_row.l_sph:.2f}D)")
                    # Note: Phoropter handles this automatically, we just track it

                if reversal:
                    return self._transition_to_duochrome_left()
                
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
            

        return self._build_response()
    
    def _process_duochrome_right(self, intent: str) -> Dict:
        """Process duochrome test for right eye.
        
        Duochrome logic:
        - Red selected → RAM (Red Add Minus) → JCC decrease, SPH -= 0.25
        - Green selected → GAP (Green Add Plus) → JCC increase, SPH += 0.25
        - Both Same → Complete process (move to next phase)
        - Reversal → Complete process (move to next phase)
        """
        if intent == "Red":
            # RAM: Red Add Minus - decrease SPH
            reversal = self._record_duochrome_choice("red")
            self.jcc_control("decrease")  # Phoropter decreases SPH by 0.25D
            self.current_row = self._copy_row_state()
            self.current_row.r_sph -= 0.25
            if reversal:
                # On reversal, transition but include updated power in response
                response = self._transition_to_left_eye_refraction()
                # Re-add power to response so frontend displays updated value
                response['power'] = self._build_response()['power']
                return response
            # Stay in duochrome for another round
            return self._build_response()
            
        elif intent == "Green":
            # GAP: Green Add Plus - increase SPH
            reversal = self._record_duochrome_choice("green")
            self.jcc_control("increase")  # Phoropter increases SPH by 0.25D
            self.current_row = self._copy_row_state()
            self.current_row.r_sph += 0.25
            if reversal:
                # On reversal, transition but include updated power in response
                response = self._transition_to_left_eye_refraction()
                # Re-add power to response so frontend displays updated value
                response['power'] = self._build_response()['power']
                return response
            # Stay in duochrome for another round
            return self._build_response()
            
        elif intent == "Both Same":
            # Both Same - complete duochrome, move to next phase
            return self._transition_to_left_eye_refraction()
        
        # Default fallback
        return self._build_response()
    
    def _process_duochrome_left(self, intent: str) -> Dict:
        """Process duochrome test for left eye.
        
        Duochrome logic:
        - Red selected → RAM (Red Add Minus) → JCC decrease, SPH -= 0.25
        - Green selected → GAP (Green Add Plus) → JCC increase, SPH += 0.25
        - Both Same → Complete process (move to next phase)
        - Reversal → Complete process (move to next phase)
        """
        if intent == "Red":
            # RAM: Red Add Minus - decrease SPH
            reversal = self._record_duochrome_choice("red")
            self.jcc_control("decrease")  # Phoropter decreases SPH by 0.25D
            self.current_row = self._copy_row_state()
            self.current_row.l_sph -= 0.25
            if reversal:
                # On reversal, transition but include updated power in response
                response = self._transition_to_binocular_balance()
                # Ensure power is in response so frontend displays updated value
                if 'power' not in response:
                    response['power'] = self._build_response()['power']
                return response
            # Stay in duochrome for another round
            return self._build_response()
            
        elif intent == "Green":
            # GAP: Green Add Plus - increase SPH
            reversal = self._record_duochrome_choice("green")
            self.jcc_control("increase")  # Phoropter increases SPH by 0.25D
            self.current_row = self._copy_row_state()
            self.current_row.l_sph += 0.25
            if reversal:
                # On reversal, transition but include updated power in response
                response = self._transition_to_binocular_balance()
                # Ensure power is in response so frontend displays updated value
                if 'power' not in response:
                    response['power'] = self._build_response()['power']
                return response
            # Stay in duochrome for another round
            return self._build_response()
            
        elif intent == "Both Same":
            # Both Same - complete duochrome, move to next phase
            return self._transition_to_binocular_balance()
        
        # Default fallback
        return self._build_response()
    
    def _process_binocular_balance(self, intent: str) -> Dict:
        """Process binocular balance phase.
        
        BINO balancing logic:
        - Top is blurry [Right Eye] → Add 0.25D Sph in Left Eye
        - Bottom is blurry [Left Eye] → Add 0.25D Sph in Right Eye
        - Both are same → Test complete
        - Worse: Go to previous state → Restore previous power
        """
        if intent == "Top is blurry [Right Eye]":
            # Save current state before making changes
            self.previous_state = {
                'r_sph': self.current_row.r_sph,
                'r_cyl': self.current_row.r_cyl,
                'r_axis': self.current_row.r_axis,
                'l_sph': self.current_row.l_sph,
                'l_cyl': self.current_row.l_cyl,
                'l_axis': self.current_row.l_axis,
                'occluder_state': self.current_row.occluder_state,
                'chart_display': self.current_row.chart_display,
            }
            # Add 0.25D Sph in Left Eye
            self.current_row = self._copy_row_state()
            self.current_row.l_sph += 0.25
            self.set_power(l_sph=self.current_row.l_sph, occluder="BINO")
            # Enable "Prev State" option for next response
            self.show_prev_state_option = True
            return self._build_response()
        
        elif intent == "Bottom is blurry [Left Eye]":
            # Save current state before making changes
            self.previous_state = {
                'r_sph': self.current_row.r_sph,
                'r_cyl': self.current_row.r_cyl,
                'r_axis': self.current_row.r_axis,
                'l_sph': self.current_row.l_sph,
                'l_cyl': self.current_row.l_cyl,
                'l_axis': self.current_row.l_axis,
                'occluder_state': self.current_row.occluder_state,
                'chart_display': self.current_row.chart_display,
            }
            # Add 0.25D Sph in Right Eye
            self.current_row = self._copy_row_state()
            self.current_row.r_sph += 0.25
            self.set_power(r_sph=self.current_row.r_sph, occluder="BINO")
            # Enable "Prev State" option for next response
            self.show_prev_state_option = True
            return self._build_response()
        
        elif intent == "Both are same":
            # Test complete
            return {
                "phase": "complete",
                "status": "complete",
                "question": "Test complete!",
                "intents": [],
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
        
        elif intent == "Prev State":
            # Restore previous state
            if self.previous_state is not None:
                self.current_row = self._copy_row_from_dict(self.previous_state)
                self.set_power(
                    r_sph=self.current_row.r_sph,
                    l_sph=self.current_row.l_sph,
                    occluder="BINO"
                )
                self.previous_state = None
                self.show_prev_state_option = False
                print("✓ Restored to previous state")
            return self._build_response()
        
        # Default fallback
        return self._build_response()
    
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
    
    def _copy_row_from_dict(self, state_dict: dict) -> RowContext:
        """Copy row state from a saved state dictionary."""
        new_row = self._init_row()
        new_row.r_sph = state_dict.get('r_sph', 0.0)
        new_row.r_cyl = state_dict.get('r_cyl', 0.0)
        new_row.r_axis = state_dict.get('r_axis', 180.0)
        new_row.l_sph = state_dict.get('l_sph', 0.0)
        new_row.l_cyl = state_dict.get('l_cyl', 0.0)
        new_row.l_axis = state_dict.get('l_axis', 180.0)
        new_row.occluder_state = state_dict.get('occluder_state', 'BINO')
        new_row.chart_display = state_dict.get('chart_display', '')
        return new_row
    
    def _is_at_cyl_threshold(self, cyl_value: float) -> bool:
        """Check if cylinder value is at a -0.50D threshold.
        
        Returns True if cyl is at -0.50, -1.00, -1.50, -2.00, etc.
        These are the points where spherical equivalent compensation is applied.
        """
        # Check if cylinder is a multiple of -0.50D
        # Account for floating point precision
        return abs(cyl_value % 0.50) < 0.01 and cyl_value < -0.01
    
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
        
        response = {
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
        
        # Add chart information if in Phase A (distance vision) or Phase B (refraction phases)
        if self.current_phase == "distance_vision":
            response["chart_info"] = {
                "available_charts": self.all_charts,
                "current_index": self.current_chart_index,
                "current_chart": self.all_charts[self.current_chart_index]
            }
        elif self.current_phase in ["right_eye_refraction", "left_eye_refraction"]:
            response["chart_info"] = {
                "available_charts": self.snellen_charts,
                "current_index": self.current_chart_index,
                "current_chart": self.snellen_charts[self.current_chart_index]
            }
        
        return response
    
    def _transition_to_jcc_axis_right(self) -> Dict:
        """Transition to JCC axis refinement for right eye."""
        self.current_phase = "jcc_axis_right"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")

        self._reset_jcc_choice_tracking()
        
        # JCC chart automatically displays when entering JCC mode - no explicit call needed
        # The phoropter defaults to Flip 1 of Axis when JCC chart is shown

        self.jcc_flip_state = "flip1"
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Right_Axis_Flip1"
        self.current_row.chart_display = "jcc_chart"
        
        # Tell frontend to auto-flip after 2 seconds
        response = self._build_response()
        response['auto_flip'] = True
        response['flip_wait_seconds'] = 2
        return response
    
    def _transition_to_jcc_axis_left(self) -> Dict:
        """Transition to JCC axis refinement for left eye."""
        self.current_phase = "jcc_axis_left"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")

        self._reset_jcc_choice_tracking()
        
        # JCC chart automatically displays when entering JCC mode - no explicit call needed
        # The phoropter defaults to Flip 1 of Axis when JCC chart is shown

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

        self._reset_jcc_choice_tracking()
        self.jcc_power_zero_flip1_count = 0  # Reset counter for new phase
        
        self.jcc_flip_state = "flip1"
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Right_Power_Flip1"
        # self.current_row.chart_display = "jcc_chart"
        
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

        self._reset_jcc_choice_tracking()
        self.jcc_power_zero_flip1_count = 0  # Reset counter for new phase
        
        self.jcc_flip_state = "flip1"
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Left_Power_Flip1"
        # self.current_row.chart_display = "jcc_chart"
        
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

        self._reset_duochrome_choice_tracking()

        self._reset_jcc_choice_tracking()
        
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Left_Occluded"
        self.current_row.chart_display = "duochrome"
        
        self.set_chart("duochrome")
        
        return self._build_response()
    
    def _transition_to_duochrome_left(self) -> Dict:
        """Transition to duochrome test for left eye."""
        self.current_phase = "duochrome_left"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")

        self._reset_duochrome_choice_tracking()

        self._reset_jcc_choice_tracking()
        
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Right_Occluded"
        self.current_row.chart_display = "duochrome"
        
        self.set_chart("duochrome")
        
        return self._build_response()
    
    def _transition_to_left_eye_refraction(self) -> Dict:
        """Transition to left eye refraction.
        
        Uses vision correction API with previous state to ensure accurate click calculations
        when transitioning from right to left eye. Sets both previous and current values
        as the same to maintain current power while switching occluder.
        """
        self.current_phase = "left_eye_refraction"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        self.current_chart_index = 0  # Start with largest chart
        self.unable_read_count = 0
        
        # Get current power values
        curr_r_sph = self.current_row.r_sph
        curr_r_cyl = self.current_row.r_cyl
        curr_r_axis = self.current_row.r_axis
        curr_l_sph = self.current_row.l_sph
        curr_l_cyl = self.current_row.l_cyl
        curr_l_axis = self.current_row.l_axis
        
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "Right_Occluded"
        self.current_row.chart_display = self.snellen_charts[0]
        
        # self.set_chart(self.snellen_charts[0])
        
        # Use vision correction API with previous state
        # Set both previous and current values as the same to maintain power while switching occluder
        self.set_power_with_prev_state(
            prev_r_sph=curr_r_sph, prev_r_cyl=curr_r_cyl, prev_r_axis=curr_r_axis,
            prev_l_sph=curr_l_sph, prev_l_cyl=curr_l_cyl, prev_l_axis=curr_l_axis,
            r_sph=curr_r_sph, r_cyl=curr_r_cyl, r_axis=curr_r_axis,
            l_sph=curr_l_sph, l_cyl=curr_l_cyl, l_axis=curr_l_axis,
            prev_aux_lens="AuxLensL",  # Previous state was testing right eye (left occluded)
            aux_lens="AuxLensR"  # Now testing left eye (right occluded)
        )
        
        # Explicitly click L button to activate left eye testing mode
        self.jcc_control("L")
        
        return self._build_response()
    
    def _transition_to_binocular_balance(self) -> Dict:
        """Transition to binocular balance."""
        self.current_phase = "binocular_balance"
        print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
        
        # Reset previous state tracking
        self.previous_state = None
        self.show_prev_state_option = False
        
        self.current_row = self._copy_row_state()
        self.current_row.occluder_state = "BINO"
        self.current_row.chart_display = "bino_chart"
        
        # Display BINO chart (chart_20)
        self.set_chart("bino_chart")
        self.set_power(occluder="BINO")
        self.jcc_control("BINO")
        
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
    
    def _setup_phase(self, phase: str) -> Dict:
        """Setup phoropter for the given phase.
        
        This method is used for the "Jump to Phase" feature to directly navigate
        to any phase with correct charts, sequences, and state initialization.
        
        Returns:
            Response dict with phase state, including auto_flip flag for JCC phases
        """
        # Set current phase
        self.current_phase = phase
        print(f"\n→ Jumping to {self.phase_names.get(phase, phase)}")
        
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
        
        # Reset refraction state
        self.current_chart_index = 0
        self.unable_read_count = 0
        
        if phase == "distance_vision":
            self.set_chart("echart_400")
            self.set_power(occluder="BINO")
            self._update_state(occluder="BINO", chart="echart_400")
            
        elif phase == "right_eye_refraction":
            # Start with the first chart in the sequence (largest)
            self.current_chart_index = 0
            self.set_chart(self.snellen_charts[0])
            self.set_power(occluder="Left_Occluded")
            self._update_state(occluder="Left_Occluded", chart=self.snellen_charts[0])
            
        elif phase == "jcc_axis_right":
            # Reset JCC choice tracking
            self._reset_jcc_choice_tracking()
            self.jcc_flip_state = "flip1"
            # Display JCC chart
            self.set_chart("jcc_chart")
            # Set JCC eye mode to R (testing right eye)
            self.jcc_control("R")
            self._update_state(occluder="Right_Axis_Flip1", chart="jcc_chart")
            # Return response with auto_flip flag
            response = self._build_response()
            response['auto_flip'] = True
            response['flip_wait_seconds'] = 2
            return response
            
        elif phase == "jcc_power_right":
            # Reset JCC choice tracking and zero flip counter
            self._reset_jcc_choice_tracking()
            self.jcc_power_zero_flip1_count = 0
            self.jcc_flip_state = "flip1"
            # Display JCC chart if not already shown
            self.set_chart("jcc_chart")
            # Set JCC eye mode to R
            self.jcc_control("R")
            # Switch to power mode
            self.jcc_control("power_axis_switch")
            self._update_state(occluder="Right_Power_Flip1", chart="jcc_chart")
            # Return response with auto_flip flag
            response = self._build_response()
            response['auto_flip'] = True
            response['flip_wait_seconds'] = 2
            return response
            
        elif phase == "duochrome_right":
            # Reset duochrome choice tracking
            self._reset_duochrome_choice_tracking()
            self.set_chart("duochrome")
            self.set_power(occluder="Left_Occluded")
            self._update_state(occluder="Left_Occluded", chart="duochrome")
            
        elif phase == "left_eye_refraction":
            # Start with the first chart in the sequence (largest)
            self.current_chart_index = 0
            self.set_chart(self.snellen_charts[0])
            self.set_power(occluder="Right_Occluded")
            self._update_state(occluder="Right_Occluded", chart=self.snellen_charts[0])
            
        elif phase == "jcc_axis_left":
            # Reset JCC choice tracking
            self._reset_jcc_choice_tracking()
            self.jcc_flip_state = "flip1"
            # Display JCC chart
            self.set_chart("jcc_chart")
            # Set JCC eye mode to L (testing left eye)
            self.jcc_control("L")
            self._update_state(occluder="Left_Axis_Flip1", chart="jcc_chart")
            # Return response with auto_flip flag
            response = self._build_response()
            response['auto_flip'] = True
            response['flip_wait_seconds'] = 2
            return response
            
        elif phase == "jcc_power_left":
            # Reset JCC choice tracking and zero flip counter
            self._reset_jcc_choice_tracking()
            self.jcc_power_zero_flip1_count = 0
            self.jcc_flip_state = "flip1"
            # Display JCC chart if not already shown
            self.set_chart("jcc_chart")
            # Set JCC eye mode to L
            self.jcc_control("L")
            # Switch to power mode
            self.jcc_control("power_axis_switch")
            self._update_state(occluder="Left_Power_Flip1", chart="jcc_chart")
            # Return response with auto_flip flag
            response = self._build_response()
            response['auto_flip'] = True
            response['flip_wait_seconds'] = 2
            return response
            
        elif phase == "duochrome_left":
            # Reset duochrome choice tracking
            self._reset_duochrome_choice_tracking()
            self.set_chart("duochrome")
            self.set_power(occluder="Right_Occluded")
            self._update_state(occluder="Right_Occluded", chart="duochrome")
            
        elif phase == "binocular_balance":
            # Reset previous state tracking
            self.previous_state = None
            self.show_prev_state_option = False
            self.set_chart("bino_chart")
            self.set_power(occluder="BINO")
            self.jcc_control("BINO")
            self._update_state(occluder="BINO", chart="bino_chart")
        
        # Return standard response for non-JCC phases
        return self._build_response()


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
