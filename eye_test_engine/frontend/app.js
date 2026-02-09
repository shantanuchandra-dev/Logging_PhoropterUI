// Eye Test Engine Frontend Application
// Integrates with Flask API backend and Phoropter API

const CONFIG = {
    backendUrl: 'http://localhost:5000',
    phoropterUrl: 'https://rajasthan-royals.preprod.lenskart.com',
    phoropterId: 'phoropter-1'
};

let sessionState = {
    sessionId: null,
    currentPhase: null,
    currentChart: null,  // Track current chart to avoid duplicate setChart calls
    currentChartIndex: 0,  // Track current chart index
    availableCharts: [],  // List of available charts
    intentsLocked: false,
    responseCount: 0,
    history: []
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Eye Test Engine Frontend Loaded');
    updateStatusIndicator(false);
});

function openArPowerModal() {
    const modal = document.getElementById('arPowerModal');
    if (modal) {
        modal.classList.add('active');
    }
}

function closeArPowerModal() {
    const modal = document.getElementById('arPowerModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

function parseArValue(value, fallback) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

async function applyArPower() {
    if (!sessionState.sessionId) {
        alert('Please start a test session first.');
        return;
    }

    const rightSph = parseArValue(document.getElementById('arRightSph').value, 0);
    const rightCyl = parseArValue(document.getElementById('arRightCyl').value, 0);
    const rightAxis = parseArValue(document.getElementById('arRightAxis').value, 180);
    const leftSph = parseArValue(document.getElementById('arLeftSph').value, 0);
    const leftCyl = parseArValue(document.getElementById('arLeftCyl').value, 0);
    const leftAxis = parseArValue(document.getElementById('arLeftAxis').value, 180);

    const power = {
        right: { sph: rightSph, cyl: rightCyl, axis: rightAxis },
        left: { sph: leftSph, cyl: leftCyl, axis: leftAxis }
    };

    try {
        showLoading(true);
        await setPower(power, 'BINO');
        addToHistory('AR power applied (BINO)', 'info');

        // Update UI display immediately
        document.getElementById('rightPower').textContent =
            `${rightSph.toFixed(2)} / ${rightCyl.toFixed(2)} / ${rightAxis.toFixed(0)}°`;
        document.getElementById('leftPower').textContent =
            `${leftSph.toFixed(2)} / ${leftCyl.toFixed(2)} / ${leftAxis.toFixed(0)}°`;
        document.getElementById('occluderState').textContent = 'BINO';

        closeArPowerModal();
    } catch (error) {
        console.error('Error applying AR power:', error);
        alert('Failed to apply AR power. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Start Test
async function startTest() {
    const btn = document.getElementById('startTestBtn');
    if (btn) btn.disabled = true;
    
    try {
        showLoading(true);
        
        // Generate session ID
        const sessionId = 'session_' + Date.now();
        sessionState.sessionId = sessionId;
        sessionState.currentChart = null;  // Reset chart tracking for new session
        
        // Reset phoropter
        await resetPhoropter();
        
        // Start session with backend
        const response = await fetch(`${CONFIG.backendUrl}/api/session/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        if (!response.ok) {
            throw new Error('Failed to start session');
        }
        
        const data = await response.json();
        
        // Update UI
        document.getElementById('welcomeScreen').style.display = 'none';
        document.getElementById('testScreen').style.display = 'block';
        
        updateSessionInfo(data);
        displayQuestion(data);
        
        // Set phoropter for first phase
        await setPhoropter(data);
        
        addToHistory('Test started', 'success');
        updateStatusIndicator(true);
        
        // Check if auto-flip is needed (JCC Flip1 → Flip2)
        if (data.auto_flip) {
            await handleAutoFlip(data.flip_wait_seconds || 2);
        }
        
    } catch (error) {
        console.error('Error starting test:', error);
        alert('Failed to start test. Make sure the backend server is running on port 5000.');
        if (btn) btn.disabled = false;
    } finally {
        showLoading(false);
    }
}

// Submit Intent Response
async function submitIntent(intent) {
    if (sessionState.intentsLocked) {
        return;
    }
    try {
        showLoading(true);
        sessionState.intentsLocked = true;
        
        // Hide all intent buttons during processing
        const intentButtonsContainer = document.getElementById('intentButtons');
        intentButtonsContainer.innerHTML = '<div class="alert alert-info">Processing...</div>';
        
        // Record response
        sessionState.responseCount++;
        addToHistory(`Response: ${intent}`, 'info');
        
        // Send to backend
        const response = await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/respond`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ intent: intent })
        });
        
        if (!response.ok) {
            throw new Error('Failed to submit response');
        }
        
        const data = await response.json();
        
        // Check if test is complete
        if (data.phase === 'complete' || data.status === 'complete') {
            await completeTest();
            return;
        }
        
        // Update UI for next question
        updateSessionInfo(data);
        
        // Update phoropter first
        await setPhoropter(data);
        
        // Display question and intents AFTER processing is complete
        displayQuestion(data);
        
        // Check if auto-flip is needed (JCC Flip1 → Flip2)
        if (data.auto_flip) {
            await handleAutoFlip(data.flip_wait_seconds || 2);
        }
        
    } catch (error) {
        console.error('Error submitting intent:', error);
        alert('Failed to submit response. Please try again.');
        sessionState.intentsLocked = false;
        // Restore intents on error
        const intentButtons = document.querySelectorAll('.intent-button');
        intentButtons.forEach(btn => btn.disabled = false);
    } finally {
        showLoading(false);
    }
}

// Handle Automatic Flip (Flip1 → wait → Flip2)
async function handleAutoFlip(waitSeconds) {
    try {
        // Hide intent buttons during auto-flip countdown
        const intentButtonsContainer = document.getElementById('intentButtons');
        const originalContent = intentButtonsContainer.innerHTML;
        intentButtonsContainer.innerHTML = '';
        
        // Show countdown in question box
        const questionBox = document.querySelector('.question-box');
        const countdownDiv = document.createElement('div');
        countdownDiv.id = 'flipCountdown';
        countdownDiv.style.cssText = 'background: #fff3e0; padding: 15px; margin-top: 15px; border-radius: 5px; text-align: center; font-size: 1.2em; color: #f57c00; font-weight: bold;';
        questionBox.appendChild(countdownDiv);
        
        // Countdown timer
        for (let i = waitSeconds; i > 0; i--) {
            countdownDiv.textContent = `⏱️ Showing Flip 2 in ${i} second${i > 1 ? 's' : ''}...`;
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        countdownDiv.textContent = '⏱️ Now showing Flip 2...';
        
        // Call backend with AUTO_FLIP to show Flip2
        const response = await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/respond`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ intent: 'AUTO_FLIP' })
        });
        
        if (!response.ok) {
            throw new Error('Failed to auto-flip');
        }
        
        const data = await response.json();
        
        // Remove countdown
        if (countdownDiv.parentNode) {
            countdownDiv.parentNode.removeChild(countdownDiv);
        }
        
        // Update UI with Flip2 state
        updateSessionInfo(data);
        displayQuestion(data);
        
        // Note: displayQuestion() creates fresh enabled buttons
        
        addToHistory('Flip 2 displayed', 'info');
        
    } catch (error) {
        console.error('Error during auto-flip:', error);
        alert('Failed to show Flip 2. Please try again.');
    }
}

// Display Question and Intents
function displayQuestion(data) {
    // Update phase badge
    const phaseName = data.phase || 'Unknown Phase';
    document.getElementById('phaseBadge').textContent = phaseName;
    
    // Update question
    const question = data.question || 'Please describe what you see.';
    document.getElementById('questionText').textContent = question;
    
    // Update chart selector visibility and content
    updateChartSelector(data);
    
    // Update intents
    const intents = data.intents || [];
    const intentButtons = document.getElementById('intentButtons');
    intentButtons.innerHTML = '';
    sessionState.intentsLocked = false;
    
    // If no intents (Flip1 state), show waiting message
    if (intents.length === 0 && data.auto_flip) {
        const waitingMsg = document.createElement('div');
        waitingMsg.className = 'alert alert-info';
        waitingMsg.textContent = 'Please observe Flip 1. Flip 2 will show automatically...';
        intentButtons.appendChild(waitingMsg);
        return;
    }
    
    intents.forEach((intent, index) => {
        const button = document.createElement('button');
        button.className = 'intent-button';
        button.textContent = `${index + 1}. ${intent}`;
        button.onclick = () => submitIntent(intent);
        intentButtons.appendChild(button);
    });
}

// Update Chart Selector
function updateChartSelector(data) {
    const chartSelector = document.getElementById('chartSelector');
    const chartGrid = document.getElementById('chartGrid');
    
    // Check if we're in Phase B (right or left eye refraction)
    const isPhaseB = data.phase && (
        data.phase.includes('Right Eye Refraction') || 
        data.phase.includes('Left Eye Refraction')
    );
    
    if (isPhaseB && data.chart_info) {
        // Show chart selector
        chartSelector.classList.add('active');
        
        // Update session state
        sessionState.availableCharts = data.chart_info.available_charts || [];
        sessionState.currentChartIndex = data.chart_info.current_index || 0;
        
        // Build chart grid
        chartGrid.innerHTML = '';
        sessionState.availableCharts.forEach((chart, index) => {
            const button = document.createElement('button');
            button.className = 'chart-button';
            if (index === sessionState.currentChartIndex) {
                button.classList.add('active');
            }
            
            const chartName = document.createElement('div');
            chartName.className = 'chart-name';
            chartName.textContent = formatChartName(chart);
            
            const chartSize = document.createElement('div');
            chartSize.className = 'chart-size';
            chartSize.textContent = extractChartSize(chart);
            
            button.appendChild(chartName);
            button.appendChild(chartSize);
            button.onclick = () => switchChart(index);
            
            chartGrid.appendChild(button);
        });
    } else {
        // Hide chart selector for other phases
        chartSelector.classList.remove('active');
    }
}

// Format chart name for display
function formatChartName(chartName) {
    // Convert "snellen_chart_200_150" to "Chart 200/150"
    const match = chartName.match(/snellen_chart_(.+)/);
    if (match) {
        return `Chart ${match[1].replace(/_/g, '/')}`;
    }
    return chartName;
}

// Extract chart size for display
function extractChartSize(chartName) {
    // Convert "snellen_chart_200_150" to "20/200 - 20/150"
    const match = chartName.match(/snellen_chart_(\d+)_(\d+)(?:_(\d+))?/);
    if (match) {
        if (match[3]) {
            return `20/${match[1]} - 20/${match[2]} - 20/${match[3]}`;
        }
        return `20/${match[1]} - 20/${match[2]}`;
    }
    return '';
}

// Switch to a different chart
async function switchChart(chartIndex) {
    if (!sessionState.sessionId) {
        alert('No active session');
        return;
    }
    
    if (chartIndex === sessionState.currentChartIndex) {
        return; // Already on this chart
    }
    
    try {
        showLoading(true);
        
        const response = await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/switch-chart`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chart_index: chartIndex })
        });
        
        if (!response.ok) {
            throw new Error('Failed to switch chart');
        }
        
        const data = await response.json();
        
        // Update phoropter
        await setPhoropter(data);
        
        // Update UI
        updateSessionInfo(data);
        displayQuestion(data);
        
        addToHistory(`Switched to chart ${chartIndex + 1}`, 'info');
        
    } catch (error) {
        console.error('Error switching chart:', error);
        alert('Failed to switch chart. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Update Session Info Panel
function updateSessionInfo(data) {
    document.getElementById('sessionId').textContent = sessionState.sessionId;
    document.getElementById('sessionStatus').textContent = 'Active';
    document.getElementById('currentPhase').textContent = data.phase || '-';
    document.getElementById('responseCount').textContent = sessionState.responseCount;
    
    // Update power info
    if (data.power) {
        const right = data.power.right || { sph: 0, cyl: 0, axis: 180 };
        const left = data.power.left || { sph: 0, cyl: 0, axis: 180 };
        
        document.getElementById('rightPower').textContent = 
            `${right.sph.toFixed(2)} / ${right.cyl.toFixed(2)} / ${right.axis.toFixed(0)}°`;
        document.getElementById('leftPower').textContent = 
            `${left.sph.toFixed(2)} / ${left.cyl.toFixed(2)} / ${left.axis.toFixed(0)}°`;
    }
    
    // Update occluder and chart
    document.getElementById('occluderState').textContent = data.occluder || 'BINO';
    document.getElementById('chartDisplay').textContent = data.chart || '-';
    
    sessionState.currentPhase = data.phase;
}

// Phoropter Control Functions
async function resetPhoropter() {
    try {
        const response = await fetch(`${CONFIG.phoropterUrl}/phoropter/${CONFIG.phoropterId}/reset`, {
            method: 'POST'
        });
        
        if (response.ok) {
            addToHistory('Phoropter reset to 0/0/180', 'success');
        }
    } catch (error) {
        console.error('Error resetting phoropter:', error);
        addToHistory('Warning: Could not reset phoropter', 'warning');
    }
}

async function setPhoropter(data) {
    try {
        // Set chart only if it has changed (avoids duplicate JCC chart calls during flip cycles)
        if (data.chart && data.chart !== sessionState.currentChart) {
            await setChart(data.chart);
            sessionState.currentChart = data.chart;
        }
        
        // Set power and occluder (skip for JCC and duochrome phases - phoropter handles internally)
        const phaseText = (data.phase || '').toLowerCase();
        const isJccPhase = phaseText.includes('jcc') || data.chart === 'jcc_chart';
        const isDuochromePhase = phaseText.includes('duochrome') || data.chart === 'duochrome';
        if (data.power && !isJccPhase && !isDuochromePhase) {
            await setPower(data.power, data.occluder);
        }
        
    } catch (error) {
        console.error('Error setting phoropter:', error);
        addToHistory('Warning: Could not update phoropter', 'warning');
    }
}

async function setChart(chartName) {
    const chartMap = {
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
    };
    
    const chartId = chartMap[chartName];
    if (!chartId) return;
    
    const payload = {
        test_cases: [{
            chart: {
                tab: "Chart1",
                chart_items: [chartId]
            }
        }]
    };
    
    await fetch(`${CONFIG.phoropterUrl}/phoropter/${CONFIG.phoropterId}/run-tests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    
    addToHistory(`Chart: ${chartName}`, 'info');
}

async function setPower(power, occluder) {
    const right = power.right || { sph: 0, cyl: 0, axis: 180 };
    const left = power.left || { sph: 0, cyl: 0, axis: 180 };
    
    // Map occluder
    let auxLens = "OFF";
    if (occluder === "Left_Occluded") {
        auxLens = "AuxLensL";
    } else if (occluder === "Right_Occluded") {
        auxLens = "AuxLensR";
    }
    
    const payload = {
        test_cases: [{
            aux_lens: auxLens,
            right_eye: {
                sph: right.sph,
                cyl: right.cyl,
                axis: right.axis
            },
            left_eye: {
                sph: left.sph,
                cyl: left.cyl,
                axis: left.axis
            }
        }]
    };
    
    await fetch(`${CONFIG.phoropterUrl}/phoropter/${CONFIG.phoropterId}/run-tests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    
    addToHistory(`Power updated - Occluder: ${occluder}`, 'info');
}

// Complete Test
async function completeTest() {
    try {
        const response = await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/end`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to end session');
        }
        
        const data = await response.json();
        
        // Hide test screen
        document.getElementById('testScreen').style.display = 'none';
        document.getElementById('completeScreen').style.display = 'block';
        
        // Display final prescription
        if (data.final_prescription) {
            const rx = data.final_prescription;
            const prescriptionHtml = `
                <div class="info-section">
                    <h4>Final Prescription</h4>
                    <div class="info-item">
                        <span class="info-label">Right Eye (OD)</span>
                        <span class="info-value">
                            SPH: ${rx.right_eye.sph.toFixed(2)} | 
                            CYL: ${rx.right_eye.cyl.toFixed(2)} | 
                            AXIS: ${rx.right_eye.axis.toFixed(0)}°
                        </span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Left Eye (OS)</span>
                        <span class="info-value">
                            SPH: ${rx.left_eye.sph.toFixed(2)} | 
                            CYL: ${rx.left_eye.cyl.toFixed(2)} | 
                            AXIS: ${rx.left_eye.axis.toFixed(0)}°
                        </span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Total Responses</span>
                        <span class="info-value">${data.total_rows || sessionState.responseCount}</span>
                    </div>
                </div>
            `;
            document.getElementById('finalPrescription').innerHTML = prescriptionHtml;
        }
        
        updateStatusIndicator(false);
        document.getElementById('sessionStatus').textContent = 'Completed';
        addToHistory('Test completed successfully', 'success');
        
    } catch (error) {
        console.error('Error completing test:', error);
        alert('Failed to complete test properly.');
    }
}

// End Test Early
async function endTest() {
    if (confirm('Are you sure you want to end the test?')) {
        await completeTest();
    }
}

// UI Helper Functions
function showLoading(show) {
    const loader = document.getElementById('loadingIndicator');
    if (show) {
        loader.classList.add('active');
    } else {
        loader.classList.remove('active');
    }
}

function updateStatusIndicator(active) {
    const indicator = document.getElementById('statusIndicator');
    if (active) {
        indicator.classList.add('status-active');
        indicator.classList.remove('status-inactive');
    } else {
        indicator.classList.add('status-inactive');
        indicator.classList.remove('status-active');
    }
}

function addToHistory(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const historyLog = document.getElementById('historyLog');
    
    // Clear "no history" message
    if (sessionState.history.length === 0) {
        historyLog.innerHTML = '';
    }
    
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `<strong>${timestamp}</strong> - ${message}`;
    
    historyLog.insertBefore(item, historyLog.firstChild);
    sessionState.history.push({ timestamp, message, type });
    
    // Keep only last 20 items
    while (historyLog.children.length > 20) {
        historyLog.removeChild(historyLog.lastChild);
    }
}

// Jump to Phase
async function jumpToPhase() {
    const select = document.getElementById('phaseSelect');
    const jumpBtn = document.getElementById('jumpBtn');
    const targetPhase = select.value;
    
    if (!targetPhase) {
        alert('Please select a phase');
        return;
    }
    
    if (!sessionState.sessionId) {
        alert('Please start a test session first');
        return;
    }
    
    jumpBtn.disabled = true;
    
    try {
        showLoading(true);
        
        const response = await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/jump`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phase: targetPhase })
        });
        
        if (!response.ok) {
            throw new Error('Failed to jump to phase');
        }
        
        const data = await response.json();
        
        // Update UI
        updateSessionInfo(data);
        displayQuestion(data);
        
        // Show test screen if not visible
        document.getElementById('welcomeScreen').style.display = 'none';
        document.getElementById('testScreen').style.display = 'block';
        
        addToHistory(`Jumped to ${data.phase}`, 'info');
        
        // If auto_flip is requested, start countdown
        if (data.auto_flip) {
            await handleAutoFlip(data.flip_wait_seconds || 2);
        }
        
        showLoading(false);
        
    } catch (error) {
        console.error('Error jumping to phase:', error);
        alert('Failed to jump to phase. Please try again.');
        showLoading(false);
    } finally {
        jumpBtn.disabled = false;
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Number keys 1-9 to select intents
    if (e.key >= '1' && e.key <= '9') {
        const index = parseInt(e.key) - 1;
        const buttons = document.querySelectorAll('.intent-button');
        if (buttons[index]) {
            buttons[index].click();
        }
    }
});
