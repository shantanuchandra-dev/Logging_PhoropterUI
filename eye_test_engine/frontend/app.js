// Eye Test Engine Frontend Application
// Integrates with Flask API backend and Phoropter API

const CONFIG = {
    backendUrl: 'http://localhost:5050',
    phoropterUrl: 'https://rajasthan-royals.preprod.lenskart.com',
    get phoropterId() {
        const el = document.getElementById('phoropterIdInput');
        return (el && el.value.trim()) ? el.value.trim() : 'phoropter-1';
    }
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

// Stored power values
let storedPower = {
    ar: null,  // {right: {sph, cyl, axis}, left: {sph, cyl, axis}}
    lenso: null  // {right: {sph, cyl, axis}, left: {sph, cyl, axis}}
};

let currentAppliedPower = 'none';  // 'none', 'ar', or 'lenso'

// Stored phoropter state snapshot for comparison
let storedPhoropterState = null;

// Optotype mapping for VA charts (Chart 1)
const OPTOTYPE_MAP = {
    "snellen_chart_200_150": ["200", "150"],
    "snellen_chart_100_80": ["100", "80"],
    "snellen_chart_70_60_50": ["70", "60", "50"],
    "snellen_chart_40_30_25": ["40", "30", "25"],
    "snellen_chart_20_15_10": ["20", "15", "10"],
    "snellen_chart_20_20_20": ["20_1", "20_2", "20_3"],
    "snellen_chart_25_20_15": ["25", "20", "15"],
    "bino_chart": ["R", "L"]
};

let currentOptotype = null;

function savePhoropterId() {
    const el = document.getElementById('phoropterIdInput');
    if (!el) return;
    const id = el.value.trim() || 'phoropter-1';
    el.value = id;
    localStorage.setItem('phoropterId', id);
    console.log(`Phoropter ID set to: ${id}`);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Eye Test Engine Frontend Loaded');

    // Restore saved phoropter ID from localStorage
    const savedId = localStorage.getItem('phoropterId');
    const idInput = document.getElementById('phoropterIdInput');
    if (idInput && savedId) {
        idInput.value = savedId;
    }

    updateStatusIndicator(false);
    populateDirectCommands();
    bindTableInteractions();
});

// ── Manual Refraction Adjustments ─────────────────────

function bindTableInteractions() {
    const table = document.getElementById('refractionTable');
    if (table) {
        // Disable context menu on the refraction area to allow right-click subtraction/addition
        table.oncontextmenu = function (e) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        };
        // Add mousedown listeners to all editable cells
        table.querySelectorAll('.rt-val').forEach(el => {
            el.addEventListener('mousedown', (event) => handleTableMousedown(event, el));
        });
    }
}

async function handleTableMousedown(event, el) {
    if (!sessionState.sessionId) {
        alert('Please start a test session first.');
        return;
    }

    // button 0 = left, button 2 = right
    const action = (event.button === 0) ? 'subtract' : (event.button === 2 ? 'add' : null);
    if (!action) return;

    // Block default behavior
    event.preventDefault();
    event.stopPropagation();

    const param = el.dataset.param; // sph, cyl, axis
    const eye = el.dataset.eye; // R or L

    let delta = 0.25;
    if (param === 'axis') delta = 5;

    // Left click subtracts, Right click adds
    if (action === 'subtract') delta = -delta;

    await applyManualPowerChange(eye, param, delta);
}

async function applyManualPowerChange(eye, param, delta) {
    if (!sessionState.lastResponse || !sessionState.lastResponse.power) return;

    // Ensure we have a working power object
    const p = sessionState.lastResponse.power;
    const eyeKey = eye === 'R' ? 'right' : 'left';

    if (!p[eyeKey]) {
        p[eyeKey] = { sph: 0, cyl: 0, axis: 180 };
    }

    let current = parseFloat(p[eyeKey][param]) || 0;
    let newVal = current + delta;

    if (param === 'axis') {
        newVal = (Math.round(newVal) % 180);
        if (newVal <= 0) newVal += 180;
    }

    p[eyeKey][param] = newVal;

    // Update UI optimistically
    updateSessionInfo(sessionState.lastResponse);

    // Call backend to update phoropter
    try {
        showLoading(true);
        // Note: The /api/session/.../set-power endpoint might not exist securely mapped for a partial update.
        // We will call setPower directly from app.js to the phoropter using the existing wrapper
        const reqPower = {
            right: {
                sph: p.right.sph || 0,
                cyl: p.right.cyl || 0,
                axis: p.right.axis || 180
            },
            left: {
                sph: p.left.sph || 0,
                cyl: p.left.cyl || 0,
                axis: p.left.axis || 180
            }
        };

        const currentOccluder = document.getElementById('occluderState').textContent || 'BINO';

        // Sync the manually adjusted power with the backend session state
        // so that the next test question calculates based on this new power.
        if (sessionState.sessionId) {
            try {
                await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/sync-power`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(reqPower)
                });
            } catch (syncErr) {
                console.warn('Failed to sync manual power to backend:', syncErr);
            }
        }

        await setPower(reqPower, currentOccluder);

        addToHistory(`Manual Adjust: ${param.toUpperCase()} ${delta > 0 ? '+' : ''}${delta} [${eye}]`, 'adjust');
    } catch (error) {
        console.error('Error applying manual power:', error);
        alert('Failed to push manual power to phoropter. Try again.');
    } finally {
        showLoading(false);
    }
}

// ── Modals & Flow ─────────────────────────────────────

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

function openLensoPowerModal() {
    const modal = document.getElementById('lensoPowerModal');
    if (modal) {
        modal.classList.add('active');
    }
}

function closeLensoPowerModal() {
    const modal = document.getElementById('lensoPowerModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

function parseArValue(value, fallback) {
    if (value === '' || value === null || value === undefined) {
        return fallback;
    }
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function saveArPower() {
    const rightSph = parseArValue(document.getElementById('arRightSph').value, null);
    const rightCyl = parseArValue(document.getElementById('arRightCyl').value, null);
    const rightAxis = parseArValue(document.getElementById('arRightAxis').value, null);
    const leftSph = parseArValue(document.getElementById('arLeftSph').value, null);
    const leftCyl = parseArValue(document.getElementById('arLeftCyl').value, null);
    const leftAxis = parseArValue(document.getElementById('arLeftAxis').value, null);

    // Check if all values are provided for both eyes
    const rightComplete = rightSph !== null && rightCyl !== null && rightAxis !== null;
    const leftComplete = leftSph !== null && leftCyl !== null && leftAxis !== null;

    if (!rightComplete || !leftComplete) {
        alert('Please enter complete power values for both eyes (SPH, CYL, AXIS).');
        return;
    }

    // Store AR power
    storedPower.ar = {
        right: { sph: rightSph, cyl: rightCyl, axis: rightAxis },
        left: { sph: leftSph, cyl: leftCyl, axis: leftAxis }
    };

    // Enable AR button
    document.getElementById('applyArBtn').disabled = false;
    document.getElementById('applyArBtn').title = 'Apply AR Power';

    addToHistory('AR power values saved', 'info');
    closeArPowerModal();
}

function saveLensoPower() {
    const rightSph = parseArValue(document.getElementById('lensoRightSph').value, null);
    const rightCyl = parseArValue(document.getElementById('lensoRightCyl').value, null);
    const rightAxis = parseArValue(document.getElementById('lensoRightAxis').value, null);
    const leftSph = parseArValue(document.getElementById('lensoLeftSph').value, null);
    const leftCyl = parseArValue(document.getElementById('lensoLeftCyl').value, null);
    const leftAxis = parseArValue(document.getElementById('lensoLeftAxis').value, null);

    // Check if all values are provided for both eyes
    const rightComplete = rightSph !== null && rightCyl !== null && rightAxis !== null;
    const leftComplete = leftSph !== null && leftCyl !== null && leftAxis !== null;

    if (!rightComplete || !leftComplete) {
        alert('Please enter complete power values for both eyes (SPH, CYL, AXIS).');
        return;
    }

    // Store Lenso power
    storedPower.lenso = {
        right: { sph: rightSph, cyl: rightCyl, axis: rightAxis },
        left: { sph: leftSph, cyl: leftCyl, axis: leftAxis }
    };

    // Enable Lenso button
    document.getElementById('applyLensoBtn').disabled = false;
    document.getElementById('applyLensoBtn').title = 'Apply Lenso Power';

    addToHistory('Lenso power values saved', 'info');
    closeLensoPowerModal();
}

function updateLocalPhoropterState(partial) {
    const base = sessionState.lastResponse || {};
    sessionState.lastResponse = {
        ...base,
        ...partial,
        power: partial.power || base.power,
        occluder: partial.occluder !== undefined ? partial.occluder : base.occluder,
        chart: partial.chart !== undefined ? partial.chart : base.chart,
        phase: partial.phase !== undefined ? partial.phase : base.phase
    };
}

function formatStateTooltip(state) {
    if (!state || !state.power) {
        return 'No state stored';
    }
    const right = state.power.right || { sph: 0, cyl: 0, axis: 180 };
    const left = state.power.left || { sph: 0, cyl: 0, axis: 180 };
    const phase = state.phase || 'Unknown';
    const chart = state.chart || 'Unknown';
    const occluder = state.occluder || 'Unknown';
    return [
        `Phase: ${phase}`,
        `Chart: ${chart}`,
        `Occluder: ${occluder}`,
        `Right: ${right.sph.toFixed(2)} / ${right.cyl.toFixed(2)} / ${right.axis.toFixed(0)}°`,
        `Left: ${left.sph.toFixed(2)} / ${left.cyl.toFixed(2)} / ${left.axis.toFixed(0)}°`
    ].join('\n');
}

function storeCompareState() {
    if (!sessionState.sessionId) {
        alert('Please start a test session first.');
        return;
    }

    const currentState = sessionState.lastResponse;
    if (!currentState || !currentState.power) {
        alert('Current phoropter state is not available yet.');
        return;
    }

    storedPhoropterState = {
        phase: currentState.phase,
        chart: currentState.chart,
        occluder: currentState.occluder,
        power: currentState.power
    };

    const btn = document.getElementById('compareStateBtn');
    if (btn) {
        btn.title = formatStateTooltip(storedPhoropterState);
    }

    addToHistory('Stored compare state', 'info');
}

async function applyStoredPower(type) {
    if (!sessionState.sessionId) {
        alert('Please start a test session first.');
        return;
    }

    // Get stored power
    const power = type === 'ar' ? storedPower.ar : storedPower.lenso;

    if (!power) {
        alert(`No ${type.toUpperCase()} power values stored. Please set them first.`);
        return;
    }

    try {
        showLoading(true);
        await setPower(power, 'BINO');

        currentAppliedPower = type;
        updatePowerButtonStates(type);

        const label = type === 'ar' ? 'AR' : 'Lenso';
        addToHistory(`${label} power applied`, 'info');

        // Update UI display
        document.getElementById('rightPower').textContent =
            `${power.right.sph.toFixed(2)} / ${power.right.cyl.toFixed(2)} / ${power.right.axis.toFixed(0)}°`;
        document.getElementById('leftPower').textContent =
            `${power.left.sph.toFixed(2)} / ${power.left.cyl.toFixed(2)} / ${power.left.axis.toFixed(0)}°`;
        document.getElementById('occluderState').textContent = 'BINO';
        updateLocalPhoropterState({ power: power, occluder: 'BINO' });
    } catch (error) {
        console.error(`Error applying ${type} power:`, error);
        alert(`Failed to apply ${type.toUpperCase()} power. Please try again.`);
    } finally {
        showLoading(false);
    }
}

function updatePowerButtonStates(activeType) {
    const arBtn = document.getElementById('applyArBtn');
    const lensoBtn = document.getElementById('applyLensoBtn');

    // Remove active class from all
    arBtn.classList.remove('active');
    lensoBtn.classList.remove('active');

    // Add active class to selected
    if (activeType === 'ar') {
        arBtn.classList.add('active');
    } else if (activeType === 'lenso') {
        lensoBtn.classList.add('active');
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
            body: JSON.stringify({ session_id: sessionId, phoropter_id: CONFIG.phoropterId })
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
        alert(`Failed to start test. Make sure the backend server is running at ${CONFIG.backendUrl}.\n\nRun: cd eye_test_engine && python api_server.py`);
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

    // Update optotype selector
    updateOptotypeSelector(data);

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

    // Check if we're in Phase A (distance vision) or Phase B (right or left eye refraction)
    const isPhaseA = data.phase && data.phase.includes('Distance Vision');
    const isPhaseB = data.phase && (
        data.phase.includes('Right Eye Refraction') ||
        data.phase.includes('Left Eye Refraction')
    );

    if ((isPhaseA || isPhaseB) && data.chart_info) {
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

// Update Optotype Selector
function updateOptotypeSelector(data, forceShow = false) {
    const optotypeSelector = document.getElementById('optotypeSelector');
    const optotypeGrid = document.getElementById('optotypeGrid');

    // Get chart from data or current session state
    const currentChartName = data.chart || sessionState.availableCharts[sessionState.currentChartIndex];

    // Check if we're in Phase A or B where optotypes are supported
    const isPhaseA = data.phase && data.phase.includes('Distance Vision');
    const isPhaseB = data.phase && (
        data.phase.includes('Right Eye Refraction') ||
        data.phase.includes('Left Eye Refraction')
    );

    const availableOptotypes = OPTOTYPE_MAP[currentChartName];

    if (forceShow || ((isPhaseA || isPhaseB) && availableOptotypes)) {
        optotypeSelector.classList.add('active');
        optotypeGrid.innerHTML = '';

        if (availableOptotypes) {
            availableOptotypes.forEach(optotype => {
                const button = document.createElement('button');
                button.className = 'optotype-button';
                if (optotype === currentOptotype) {
                    button.classList.add('active');
                }
                button.textContent = optotype;
                button.onclick = () => switchOptotype(optotype);
                optotypeGrid.appendChild(button);
            });
        } else {
            optotypeGrid.innerHTML = '<div style="font-size: 0.9em; color: #666; padding: 10px;">No specific optotypes for this chart</div>';
        }
    } else {
        optotypeSelector.classList.remove('active');
    }
}

// Populate Direct Chart Commands
function populateDirectCommands() {
    const container = document.getElementById('directCommands');
    if (!container) return;

    container.innerHTML = `
        <div class="command-group">
            <select id="directChartSelect" class="phase-jump select" style="width: 100%; min-width: 0; border-color: #667eea; margin-bottom: 12px; height: 38px; padding: 4px 10px; font-size: 0.95em;">
                <option value="">-- Choose Chart --</option>
            </select>
            <div id="directOptotypeGrid" class="optotype-grid" style="grid-template-columns: repeat(auto-fit, minmax(60px, 1fr)); gap: 8px;">
                <!-- Buttons populated based on selection -->
            </div>
        </div>
    `;

    const select = document.getElementById('directChartSelect');
    const grid = document.getElementById('directOptotypeGrid');

    const chartGroups = [
        { id: "snellen_chart_200_150", label: "Chart 200/150", optotypes: ["200", "150"] },
        { id: "snellen_chart_100_80", label: "Chart 100/80", optotypes: ["100", "80"] },
        { id: "snellen_chart_70_60_50", label: "Chart 70/60/50", optotypes: ["70", "60", "50"] },
        { id: "snellen_chart_40_30_25", label: "Chart 40/30/25", optotypes: ["40", "30", "25"] },
        { id: "snellen_chart_20_15_10", label: "Chart 20/15/10", optotypes: ["20", "15", "10"] },
        { id: "snellen_chart_20_20_20", label: "Chart 20/20 (Cols)", optotypes: ["20_1", "20_2", "20_3"] },
        { id: "snellen_chart_25_20_15", label: "Chart 25/20/15", optotypes: ["25", "20", "15"] },
        { id: "bino_chart", label: "Chart 20 (R/L)", optotypes: ["R", "L"] }
    ];

    chartGroups.forEach(group => {
        const option = document.createElement('option');
        option.value = group.id;
        option.textContent = group.label;
        select.appendChild(option);
    });

    select.onchange = () => {
        const groupId = select.value;
        const group = chartGroups.find(g => g.id === groupId);
        grid.innerHTML = '';
        if (group) {
            group.optotypes.forEach(opt => {
                const btn = document.createElement('button');
                btn.className = 'optotype-button';
                btn.style.padding = '8px 4px';
                btn.style.fontSize = '0.9em';
                btn.textContent = opt;
                btn.onclick = () => executeDirectCommand(groupId, opt);
                grid.appendChild(btn);
            });
        }
    };
}

async function executeDirectCommand(chartName, optotype) {
    if (!sessionState.sessionId) {
        // Start a temporary session if none exists
        if (confirm('No active session. Start a quick test session?')) {
            await startTest();
        } else {
            return;
        }
    }

    try {
        showLoading(true);
        currentOptotype = optotype;

        // Find if this chart is in available charts to keep internal state in sync
        const chartIdx = sessionState.availableCharts.indexOf(chartName);
        if (chartIdx !== -1) {
            sessionState.currentChartIndex = chartIdx;
            // Update the main chart selector UI if active
            const chartButtons = document.querySelectorAll('.chart-button');
            chartButtons.forEach((btn, idx) => {
                if (idx === chartIdx) btn.classList.add('active');
                else btn.classList.remove('active');
            });
        }

        await setChart(chartName, optotype);

        // Update current optotype UI and FORCE it to show (opens UI completely)
        updateOptotypeSelector({ chart: chartName, phase: sessionState.currentPhase }, true);

        addToHistory(`Direct command: ${chartName} [${optotype}]`, 'info');
    } catch (error) {
        console.error('Error executing direct command:', error);
        alert('Failed to execute command. Please check console.');
    } finally {
        showLoading(false);
    }
}

// Update Optotype Selector
function updateOptotypeSelector(data) {
    const optotypeSelector = document.getElementById('optotypeSelector');
    const optotypeGrid = document.getElementById('optotypeGrid');
    const currentChartName = sessionState.availableCharts[sessionState.currentChartIndex];

    // Check if we're in Phase A or B where optotypes are supported
    const isPhaseA = data.phase && data.phase.includes('Distance Vision');
    const isPhaseB = data.phase && (
        data.phase.includes('Right Eye Refraction') ||
        data.phase.includes('Left Eye Refraction')
    );

    const availableOptotypes = OPTOTYPE_MAP[currentChartName];

    if ((isPhaseA || isPhaseB) && availableOptotypes) {
        optotypeSelector.classList.add('active');
        optotypeGrid.innerHTML = '';

        availableOptotypes.forEach(optotype => {
            const button = document.createElement('button');
            button.className = 'optotype-button';
            if (optotype === currentOptotype) {
                button.classList.add('active');
            }
            button.textContent = optotype;
            button.onclick = () => switchOptotype(optotype);
            optotypeGrid.appendChild(button);
        });
    } else {
        optotypeSelector.classList.remove('active');
    }
}

// Switch to a specific optotype
async function switchOptotype(optotype) {
    if (!sessionState.sessionId) {
        alert('No active session');
        return;
    }

    if (optotype === currentOptotype) {
        // Already selected, but allow re-clicking to trigger phoropter if needed
    }

    try {
        showLoading(true);
        currentOptotype = optotype;

        const chartName = sessionState.availableCharts[sessionState.currentChartIndex];
        await setChart(chartName, optotype);

        // Update UI state
        const buttons = document.querySelectorAll('.optotype-button');
        buttons.forEach(btn => {
            if (btn.textContent === optotype) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        addToHistory(`Select optotype: ${optotype}`, 'info');
    } catch (error) {
        console.error('Error switching optotype:', error);
        alert('Failed to switch optotype. Please try again.');
    } finally {
        showLoading(false);
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

        // Reset optotype when switching chart
        currentOptotype = null;

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

        // Update refraction table cells directly
        const rSphDoc = document.getElementById('rt-r-sph');
        if (rSphDoc) rSphDoc.textContent = (right.sph >= 0 ? '+' : '') + right.sph.toFixed(2);

        const rCylDoc = document.getElementById('rt-r-cyl');
        if (rCylDoc) rCylDoc.textContent = (right.cyl >= 0 ? '+' : '') + right.cyl.toFixed(2);

        const rAxisDoc = document.getElementById('rt-r-axis');
        if (rAxisDoc) rAxisDoc.textContent = right.axis.toFixed(0);

        const lSphDoc = document.getElementById('rt-l-sph');
        if (lSphDoc) lSphDoc.textContent = (left.sph >= 0 ? '+' : '') + left.sph.toFixed(2);

        const lCylDoc = document.getElementById('rt-l-cyl');
        if (lCylDoc) lCylDoc.textContent = (left.cyl >= 0 ? '+' : '') + left.cyl.toFixed(2);

        const lAxisDoc = document.getElementById('rt-l-axis');
        if (lAxisDoc) lAxisDoc.textContent = left.axis.toFixed(0);

        // Show ADD column only during near vision phases
        const phaseText = (data.phase || '').toLowerCase();
        const isNearPhase = phaseText.includes('near vision');
        const addColHeader = document.getElementById('addColHeader');
        const rAddCell = document.getElementById('rt-r-add');
        const lAddCell = document.getElementById('rt-l-add');
        const addDisplay = isNearPhase ? '' : 'none';
        if (addColHeader) addColHeader.style.display = addDisplay;
        if (rAddCell) rAddCell.style.display = addDisplay;
        if (lAddCell) lAddCell.style.display = addDisplay;

        if (isNearPhase) {
            const rAdd = (right.add || 0);
            const lAdd = (left.add || 0);
            if (rAddCell) rAddCell.textContent = '+' + rAdd.toFixed(2);
            if (lAddCell) lAddCell.textContent = '+' + lAdd.toFixed(2);
        }
    }

    // Update occluder and chart
    document.getElementById('occluderState').textContent = data.occluder || 'BINO';
    document.getElementById('chartDisplay').textContent = data.chart || '-';

    sessionState.currentPhase = data.phase;
    sessionState.lastResponse = data;
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
        if (data.chart && (data.chart !== sessionState.currentChart || currentOptotype !== null)) {
            await setChart(data.chart, currentOptotype);
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

async function setChart(chartName, optotype = null) {
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
        "bino_chart": "chart_20",
    };

    const chartId = chartMap[chartName];
    if (!chartId) return;

    const chartItems = [chartId];
    if (optotype) {
        chartItems.push(optotype);
    }

    const payload = {
        test_cases: [{
            chart: {
                tab: "Chart1",
                chart_items: chartItems
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
            const rAdd = rx.right_eye.add || 0;
            const lAdd = rx.left_eye.add || 0;
            const hasAdd = rAdd !== 0 || lAdd !== 0;
            const prescriptionHtml = `
                <div class="info-section">
                    <h4>Final Prescription</h4>
                    <div class="info-item">
                        <span class="info-label">Right Eye (OD)</span>
                        <span class="info-value">
                            SPH: ${rx.right_eye.sph.toFixed(2)} | 
                            CYL: ${rx.right_eye.cyl.toFixed(2)} | 
                            AXIS: ${rx.right_eye.axis.toFixed(0)}°${hasAdd ? ` | 
                            ADD: +${rAdd.toFixed(2)}` : ''}
                        </span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Left Eye (OS)</span>
                        <span class="info-value">
                            SPH: ${rx.left_eye.sph.toFixed(2)} | 
                            CYL: ${rx.left_eye.cyl.toFixed(2)} | 
                            AXIS: ${rx.left_eye.axis.toFixed(0)}°${hasAdd ? ` | 
                            ADD: +${lAdd.toFixed(2)}` : ''}
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
