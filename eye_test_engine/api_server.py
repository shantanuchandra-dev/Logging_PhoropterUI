#!/usr/bin/env python3
"""
Simple Flask API server for interactive eye test sessions.
"""
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import urllib.error
import urllib.parse
import urllib.request

from interactive_session import InteractiveSession

# Load io/outputs.py directly to avoid clash with Python's built-in io module
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "io_outputs",
    str(Path(__file__).resolve().parent / "io" / "outputs.py"),
)
_outputs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_outputs)
write_session_csv = _outputs.write_session_csv
write_session_metadata = _outputs.write_session_metadata
append_to_combined_log = _outputs.append_to_combined_log
append_to_combined_metadata = _outputs.append_to_combined_metadata
build_session_metadata = _outputs.build_session_metadata

app = Flask(__name__)
CORS(app)

# Global session storage (in production, use proper session management)
sessions = {}
PHOROPTER_BASE_URL = "https://rajasthan-royals.preprod.lenskart.com"

LOGS_DIR = Path(__file__).parent / "logs"
SESSIONS_DIR = LOGS_DIR / "sessions"
COMBINED_LOG_PATH = LOGS_DIR / "combined_log.csv"
COMBINED_META_PATH = LOGS_DIR / "combined_metadata.csv"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _log_api_command(action: str, payload: dict) -> None:
    """Log incoming API commands for debugging."""
    print(f"[API] {action}: {json.dumps(payload, ensure_ascii=False)}")


def _request_payload() -> dict:
    """Read JSON payload for standard and beacon requests."""
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload

    raw = request.get_data(as_text=True) or ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _proxy_request(method: str, path: str, payload: dict | None = None, query: dict | None = None):
    """Forward a request to broker API and return flask response tuple."""
    url = f"{PHOROPTER_BASE_URL}{path}"
    if query:
        cleaned_query = {k: v for k, v in query.items() if v is not None and v != ""}
        if cleaned_query:
            url = f"{url}?{urllib.parse.urlencode(cleaned_query)}"

    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8")
            if body:
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = {"raw": body}
            else:
                parsed = {"status": "success"}
            return jsonify(parsed), resp.getcode()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"error": body or str(e)}
        if "error" not in parsed and "reason" not in parsed:
            parsed["error"] = str(e)
        return jsonify(parsed), e.code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route('/api/devices', methods=['GET'])
def list_devices():
    """List devices from phoropter broker."""
    include_all = request.args.get("all")
    return _proxy_request("GET", "/devices", query={"all": include_all})


@app.route('/api/devices/<device_id>', methods=['GET'])
def get_device(device_id):
    """Get single device details."""
    return _proxy_request("GET", f"/devices/{device_id}")


@app.route('/api/devices/<device_id>/acquire', methods=['POST'])
def acquire_device(device_id):
    """Acquire device lock for a brain."""
    payload = _request_payload()
    _log_api_command(f"/api/devices/{device_id}/acquire", payload)
    return _proxy_request("POST", f"/devices/{device_id}/acquire", payload=payload)


@app.route('/api/devices/<device_id>/release', methods=['POST'])
def release_device(device_id):
    """Release device lock for a brain."""
    payload = _request_payload()
    _log_api_command(f"/api/devices/{device_id}/release", payload)
    return _proxy_request("POST", f"/devices/{device_id}/release", payload=payload)


@app.route('/api/devices/<device_id>/heartbeat', methods=['POST'])
def heartbeat_device(device_id):
    """Send heartbeat for active brain lock."""
    payload = _request_payload()
    _log_api_command(f"/api/devices/{device_id}/heartbeat", payload)
    return _proxy_request("POST", f"/devices/{device_id}/heartbeat", payload=payload)


@app.route('/api/brains', methods=['GET'])
def list_brains():
    """List active brains from broker."""
    return _proxy_request("GET", "/brains")


@app.route('/api/events', methods=['GET'])
def list_events():
    """List broker events/audit logs."""
    limit = request.args.get("limit", "20")
    return _proxy_request("GET", "/events", query={"limit": limit})


@app.route('/api/phoropter/<device_id>/sync-state', methods=['POST'])
def sync_phoropter_state(device_id):
    """Sync broker internal state without physical clicks."""
    payload = _request_payload()
    _log_api_command(f"/api/phoropter/{device_id}/sync-state", payload)
    return _proxy_request("POST", f"/phoropter/{device_id}/sync-state", payload=payload)


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

    # Record as a Manual row in session history
    delta = session._compute_change_delta("", "Manual")
    session._stamp_row(session.current_row, "Manual", delta)
    session.session_history.append(session.current_row)
    session.current_row = session._copy_row_state()
    
    return jsonify({
        "session_id": session_id,
        "status": "success"
    })



@app.route('/api/session/<session_id>/end', methods=['POST'])
def end_session(session_id):
    """End session, write CSV logs, and return final prescription."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    payload = _request_payload()
    _log_api_command(f"/api/session/{session_id}/end", payload)
    
    session = sessions[session_id]
    session.session_end_time = datetime.now()

    # AR, Lensometry, and operator name sent from frontend
    ar = payload.get("ar", None)
    lenso = payload.get("lenso", None)
    operator_name = payload.get("operator_name", "")
    
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

    # Determine all phases in the protocol to find skipped ones
    all_phase_ids = list(session.phase_names.keys())
    phases_completed = session._phases_visited
    phases_skipped = [p for p in all_phase_ids if p not in phases_completed]

    # Build and write session metadata + CSV
    try:
        metadata = build_session_metadata(
            session_id=session_id,
            phoropter_id=session.phoropter_id,
            session_start_time=session.session_start_time or datetime.now(),
            session_end_time=session.session_end_time,
            completion_status="completed",
            rows=session.session_history,
            ar=ar,
            lensometry=lenso,
            phase_jump_count=session._phase_jump_count,
            unable_to_read_count=session.unable_read_count,
            jcc_cycles_right=session.jcc_cycle_count,
            jcc_cycles_left=getattr(session, "jcc_cycle_count", 0),
            phases_completed=phases_completed,
            phases_skipped=phases_skipped,
            duration_per_phase=session.get_duration_per_phase(),
            operator_name=operator_name,
        )

        csv_path = SESSIONS_DIR / f"{session_id}.csv"
        meta_path = SESSIONS_DIR / f"{session_id}_metadata.json"

        write_session_csv(session.session_history, csv_path)
        write_session_metadata(metadata, meta_path)
        append_to_combined_log(session.session_history, session_id, COMBINED_LOG_PATH)
        append_to_combined_metadata(metadata, COMBINED_META_PATH)

        print(f"[LOG] Session {session_id}: CSV → {csv_path}, Meta → {meta_path}")
    except Exception as e:
        print(f"[LOG ERROR] Failed to write session logs: {e}")
    
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
    print("  GET  /api/devices")
    print("  GET  /api/devices/<id>")
    print("  POST /api/devices/<id>/acquire")
    print("  POST /api/devices/<id>/heartbeat")
    print("  POST /api/devices/<id>/release")
    print("  GET  /api/brains")
    print("  GET  /api/events")
    print("  POST /api/phoropter/<id>/sync-state")
    print("  POST /api/session/start")
    print("  POST /api/session/<id>/respond")
    print("  POST /api/session/<id>/jump")
    print("  POST /api/session/<id>/switch-chart")
    print("  POST /api/session/<id>/sync-power")
    print("  GET  /api/session/<id>/status")
    print("  POST /api/session/<id>/end")
    app.run(host='0.0.0.0', port=5050, debug=True)
