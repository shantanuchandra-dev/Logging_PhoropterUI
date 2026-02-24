#!/usr/bin/env python3
"""
Simple Flask API server for interactive eye test sessions.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path

from interactive_session import InteractiveSession

app = Flask(__name__)
CORS(app)

# Global session storage (in production, use proper session management)
sessions = {}


def _log_api_command(action: str, payload: dict) -> None:
    """Log incoming API commands for debugging."""
    print(f"[API] {action}: {json.dumps(payload, ensure_ascii=False)}")


@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new eye test session."""
    payload = request.json or {}
    _log_api_command("/api/session/start", payload)
    session_id = payload.get('session_id', 'default')
    phoropter_id = payload.get('phoropter_id', 'phoropter-1')
    
    # Create new session with the specified phoropter device ID
    session = InteractiveSession(phoropter_id=phoropter_id)
    sessions[session_id] = session
    
    # Start distance vision phase
    state = session.start_distance_vision()
    
    return jsonify({
        "session_id": session_id,
        "status": "started",
        **state
    })


@app.route('/api/session/<session_id>/respond', methods=['POST'])
def respond(session_id):
    """Process patient response and get next question."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    payload = request.json or {}
    _log_api_command(f"/api/session/{session_id}/respond", payload)
    intent = payload.get('intent')
    
    if not intent:
        return jsonify({"error": "Intent required"}), 400
    
    # Process response
    next_state = session.process_response(intent)
    
    return jsonify({
        "session_id": session_id,
        "status": "active",
        **next_state
    })


@app.route('/api/session/<session_id>/status', methods=['GET'])
def get_status(session_id):
    """Get current session status."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    
    return jsonify({
        "session_id": session_id,
        "current_phase": session.current_phase,
        "total_rows": len(session.session_history),
        "current_power": {
            "right": {
                "sph": session.current_row.r_sph,
                "cyl": session.current_row.r_cyl,
                "axis": session.current_row.r_axis,
                "add": getattr(session, "add_right", 0.0),
            },
            "left": {
                "sph": session.current_row.l_sph,
                "cyl": session.current_row.l_cyl,
                "axis": session.current_row.l_axis,
                "add": getattr(session, "add_left", 0.0),
            }
        }
    })


@app.route('/api/session/<session_id>/jump', methods=['POST'])
def jump_to_phase(session_id):
    """Jump directly to a specific phase."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    payload = request.json or {}
    _log_api_command(f"/api/session/{session_id}/jump", payload)
    target_phase = payload.get('phase')
    
    if not target_phase:
        return jsonify({"error": "Phase required"}), 400
    
    # Setup the target phase
    try:
        # _setup_phase now returns a response dict with all necessary state
        state = session._setup_phase(target_phase)
        
        return jsonify({
            "session_id": session_id,
            "status": "active",
            **state
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/session/<session_id>/switch-chart', methods=['POST'])
def switch_chart(session_id):
    """Switch to a different chart during refraction phase."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    payload = request.json or {}
    _log_api_command(f"/api/session/{session_id}/switch-chart", payload)
    chart_index = payload.get('chart_index')
    
    if chart_index is None:
        return jsonify({"error": "chart_index required"}), 400
    
    # Switch chart
    try:
        state = session.switch_chart(chart_index)
        
        return jsonify({
            "session_id": session_id,
            "status": "active",
            **state
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to switch chart: {str(e)}"}), 500


@app.route('/api/session/<session_id>/sync-power', methods=['POST'])
def sync_power(session_id):
    """Sync manual power changes from frontend to backend session state."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
        
    session = sessions[session_id]
    payload = request.json or {}
    _log_api_command(f"/api/session/{session_id}/sync-power", payload)
    
    right = payload.get('right', {})
    left = payload.get('left', {})
    
    # Update internal state (current_row)
    if 'sph' in right: session.current_row.r_sph = float(right['sph'])
    if 'cyl' in right: session.current_row.r_cyl = float(right['cyl'])
    if 'axis' in right: session.current_row.r_axis = float(right['axis'])
    
    if 'sph' in left: session.current_row.l_sph = float(left['sph'])
    if 'cyl' in left: session.current_row.l_cyl = float(left['cyl'])
    if 'axis' in left: session.current_row.l_axis = float(left['axis'])
    
    return jsonify({
        "session_id": session_id,
        "status": "success"
    })



@app.route('/api/session/<session_id>/end', methods=['POST'])
def end_session(session_id):
    """End session and get final prescription."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    _log_api_command(f"/api/session/{session_id}/end", {})
    
    session = sessions[session_id]
    
    # Get final prescription
    if session.session_history:
        last_row = session.session_history[-1]
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
            }
        }
    else:
        final_rx = {}
    
    # Clean up session
    del sessions[session_id]
    
    return jsonify({
        "session_id": session_id,
        "status": "ended",
        "total_rows": len(session.session_history),
        "final_prescription": final_rx
    })


if __name__ == '__main__':
    print("Starting Eye Test API Server...")
    print("Available endpoints:")
    print("  POST /api/session/start")
    print("  POST /api/session/<id>/respond")
    print("  POST /api/session/<id>/jump")
    print("  POST /api/session/<id>/switch-chart")
    print("  POST /api/session/<id>/sync-power")
    print("  GET  /api/session/<id>/status")
    print("  POST /api/session/<id>/end")
    app.run(host='0.0.0.0', port=5050, debug=True)
