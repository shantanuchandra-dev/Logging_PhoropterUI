# Eye Test Engine - Interactive Frontend

A beautiful, modern web interface for conducting interactive eye tests with automatic phoropter control.

## Features

✅ **Interactive UI** - Clean, modern interface with real-time updates  
✅ **Automatic Phoropter Control** - Sends curl commands to control the phoropter  
✅ **Phase-by-Phase Flow** - Guides through all 10 test phases  
✅ **Intent Selection** - Click or use keyboard (1-9) to select responses  
✅ **Live Status Panel** - Shows current power, occluder, chart, and history  
✅ **Final Prescription** - Displays complete prescription at the end  

---

## Quick Start

### 1. Start the Backend API Server

```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI
python -m eye_test_engine.api_server
```

The server will start on `http://localhost:5000`

### 2. Open the Frontend

Open `index.html` in your web browser:

```bash
# Option 1: Double-click index.html in Finder

# Option 2: Use Python's built-in server
cd eye_test_engine/frontend
python3 -m http.server 8080
# Then open http://localhost:8080 in your browser

# Option 3: Direct file path
open index.html
```

### 3. Start Testing!

1. Click "Start Eye Test"
2. Read the question displayed
3. Select your response from the intent options
4. The phoropter will automatically update
5. Continue through all phases
6. View your final prescription

---

## Architecture

```
┌─────────────────┐
│   Web Browser   │
│   (Frontend)    │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  Flask Server   │
│  (Backend API)  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│  State  │ │  Phoropter   │
│ Machine │ │     API      │
└─────────┘ └──────────────┘
```

### Components

1. **Frontend (HTML/JS)**
   - `index.html` - Main UI with styling
   - `app.js` - Application logic and API calls

2. **Backend (Python)**
   - `api_server.py` - Flask REST API
   - `interactive_session.py` - Session orchestrator
   - `core/state_machine.py` - Phase logic

3. **Phoropter API**
   - External API at `rajasthan-royals.preprod.lenskart.com`
   - Controls physical phoropter hardware

---

## Test Flow

### Phase Sequence

1. **Distance Vision** (BINO + E-chart)
   - Question: "Are you able to see big E clearly?"
   - Intents: Able to read, Blurry, Unable to read

2. **Right Eye Refraction** (Left Occluded + Snellen)
   - Question: "I'm covering your left eye. Please read the line..."
   - Intents: Able to read, Blurry, Unable to read, Getting better

3. **JCC Axis Right** (Right_Axis_Flip1/2 + JCC Chart)
   - Question: "Focus on the dot chart. Is this better? (Flip 1)" / "Or is this better? (Flip 2)"
   - Intents: GAP Axis, RAM Axis, Both Same

4. **JCC Power Right** (Right_Power_Flip1/2 + JCC Chart)
   - Question: "Focus on the dot chart. Is this better? (Flip 1)" / "Or is this better? (Flip 2)"
   - Intents: GAP Power, RAM Power, Both Same

5. **Duochrome Right** (Left Occluded + Duochrome)
   - Question: "Which is clearer: red or green, or are they the same?"
   - Intents: Red, Green, Both Same

6. **Left Eye Refraction** (Right Occluded + Snellen)
   - Same as Right Eye Refraction

7. **JCC Axis Left** (Left_Axis_Flip1/2 + JCC Chart)
   - Same as JCC Axis Right

8. **JCC Power Left** (Left_Power_Flip1/2 + JCC Chart)
   - Same as JCC Power Right

9. **Duochrome Left** (Right Occluded + Duochrome)
   - Same as Duochrome Right

10. **Binocular Balance** (BINO + Snellen)
    - Question: "Please read the line you can see clearly."
    - Intents: Able to read, Blurry, Unable to read

---

## UI Features

### Status Panel

- **Session Status** - Active/Inactive indicator
- **Session ID** - Unique identifier for this test
- **Current Phase** - Which phase you're in
- **Response Count** - Number of responses given

### Power Display

- **Right Eye** - SPH / CYL / AXIS
- **Left Eye** - SPH / CYL / AXIS
- **Occluder** - Current occluder state
- **Chart** - Current chart being displayed

### History Log

- Real-time log of all actions
- Timestamps for each event
- Color-coded by type (info/success/warning)

### Keyboard Shortcuts

- **1-9** - Select intent by number
- Works even when not focused on buttons

---

## API Endpoints

### Start Session
```bash
POST http://localhost:5000/api/session/start
Content-Type: application/json

{
  "session_id": "session_123"
}
```

### Submit Response
```bash
POST http://localhost:5000/api/session/session_123/respond
Content-Type: application/json

{
  "intent": "Able to read"
}
```

### Get Status
```bash
GET http://localhost:5000/api/session/session_123/status
```

### End Session
```bash
POST http://localhost:5000/api/session/session_123/end
```

---

## Customization

### Change Backend URL

Edit `app.js`:
```javascript
const CONFIG = {
    backendUrl: 'http://your-server:5000',
    phoropterUrl: 'https://your-phoropter-api.com',
    phoropterId: 'your-phoropter-id'
};
```

### Change Colors

Edit the CSS in `index.html`:
```css
/* Main gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Accent color */
color: #667eea;
```

### Add Custom Phases

Edit `interactive_session.py` to add new phases to the flow.

---

## Troubleshooting

### Backend Not Connecting

**Problem:** "Failed to start test. Make sure the backend server is running on port 5000."

**Solution:**
1. Check if Flask server is running: `python -m eye_test_engine.api_server`
2. Verify port 5000 is not in use: `lsof -i :5000`
3. Check CORS is enabled (already configured in `api_server.py`)

### Phoropter Not Responding

**Problem:** "Warning: Could not update phoropter"

**Solution:**
1. Check network access to phoropter API
2. Verify phoropter ID is correct
3. Check API endpoint URL
4. Test with curl command directly:
   ```bash
   curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/reset
   ```

### CORS Errors

**Problem:** "Access to fetch blocked by CORS policy"

**Solution:**
1. Use `python -m http.server` to serve the frontend
2. Or install Flask-CORS: `pip install flask-cors` (already included)

### Session Not Found

**Problem:** "Session not found" error

**Solution:**
1. Restart the backend server
2. Clear browser cache
3. Start a new test

---

## Development

### File Structure

```
frontend/
├── index.html          # Main UI
├── app.js             # Application logic
└── README.md          # This file
```

### Adding New Features

1. **New Phase:** Add to `_determine_next_phase()` in `interactive_session.py`
2. **New Intent:** Update `protocol.yaml` configuration
3. **New UI Element:** Add to `index.html` and update `app.js`

### Testing

1. **Manual Testing:** Use the web interface
2. **API Testing:** Use curl or Postman
3. **Demo Mode:** Run `python -m eye_test_engine.demo_conversation`

---

## Production Deployment

### Backend

```bash
# Use gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 eye_test_engine.api_server:app
```

### Frontend

```bash
# Use nginx or Apache to serve static files
# Or deploy to Netlify/Vercel/GitHub Pages
```

### Security

- Add authentication for API endpoints
- Use HTTPS for production
- Implement session timeouts
- Add rate limiting

---

## Credits

Built with:
- **Flask** - Python web framework
- **Vanilla JavaScript** - No frameworks needed!
- **Modern CSS** - Gradients and animations
- **Eye Test Engine** - Custom state machine

---

## Support

For issues or questions:
1. Check the main `README.md` in `eye_test_engine/`
2. Review `API_USAGE.md` for curl examples
3. See `QUICK_START.md` for basic usage

---

**Ready to test? Start the backend and open `index.html`!** 🚀
