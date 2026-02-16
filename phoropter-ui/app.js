/* ====================================================
   Phoropter UI — Application Logic
   State management, controls, memory, charts, JCC, export
   ==================================================== */

// Phase list (aligned with Eye Test Engine PHASE_NAMING.md)
const REFRACTION_PHASES = [
    { id: 'distance_vision', label: 'A: Distance Vision' },
    { id: 'right_eye_refraction', label: 'B: Right Eye Refraction' },
    { id: 'jcc_axis_right', label: 'E: JCC Axis Right' },
    { id: 'jcc_power_right', label: 'F: JCC Power Right' },
    { id: 'duochrome_right', label: 'G: Duochrome Right' },
    { id: 'left_eye_refraction', label: 'D: Left Eye Refraction' },
    { id: 'jcc_axis_left', label: 'H: JCC Axis Left' },
    { id: 'jcc_power_left', label: 'I: JCC Power Left' },
    { id: 'duochrome_left', label: 'J: Duochrome Left' },
    { id: 'binocular_balance', label: 'K: Binocular Balance' }
];
const JCC_PHASE_IDS = ['jcc_axis_right', 'jcc_power_right', 'jcc_axis_left', 'jcc_power_left'];

// ── Session State ──────────────────────────────────
const session = {
    patient: { name: '', age: '', engagementId: '', customerId: '' },
    branch: { code: '', staff: '' },
    ar: {
        R: { sph: '', cyl: '', axis: '', add: '', pd: '' },
        L: { sph: '', cyl: '', axis: '', add: '', pd: '' }
    },
    lenso: {
        R: { sph: '', cyl: '', axis: '', add: '' },
        L: { sph: '', cyl: '', axis: '', add: '' }
    },
    subjective: {
        R: { sph: '0.00', cyl: '0.00', axis: '180', add: '0.00', pd: '32.0' },
        L: { sph: '0.00', cyl: '0.00', axis: '180', add: '0.00', pd: '32.0' }
    },
    memorySlots: [null, null, null],
    log: [],
    activeEye: 'R',       // 'R', 'L', or 'BOTH'
    finalized: false,
    currentPhase: 'distance_vision',
    // Chart state
    activeChartTab: 1,
    activeChartIndex: 0,
    // JCC state
    jccMode: 'axis',       // 'axis' or 'power'
    jccActiveFlip: null,    // null, 1, or 2
    // Specialty state
    isPinholeActive: false
};

// ── Chart Data ─────────────────────────────────────
// Each chart tab has optotype groups. Aligned with Eye Test Engine clinical data.
const chartData = {
    1: [ // Standard Snellen (for Refraction)
        { chars: 'M W', size: '20/200', label: 'Snellen 200/150' },
        { chars: 'E N H', size: '20/100', label: 'Snellen 100/80' },
        { chars: 'H B V', size: '20/70', label: 'Snellen 70/60/50' },
        { chars: 'S L C', size: '20/40', label: 'Snellen 40/30/25' },
        { chars: 'V L N E A', size: '20/25', label: 'Snellen 25/20/15' },
        { chars: 'D A O P F C', size: '20/20', label: 'Snellen 20/20/20' },
        { chars: 'S F L C T', size: '20/15', label: 'Snellen 20/15/10' },
    ],
    2: [ // E-Charts
        { chars: 'E', size: '20/400', label: 'E-Chart 400', type: 'e-chart' },
        { chars: 'E E', size: '20/100', label: 'Tumbling E 100', type: 'e-chart' },
        { chars: 'E E E', size: '20/60', label: 'Tumbling E 60', type: 'e-chart' },
    ],
    3: [ // Specialty / Clinical
        { chars: '🟢 🔴', size: 'N/A', label: 'Duochrome', type: 'duochrome' },
        { chars: '● ● ●', size: 'N/A', label: 'JCC Chart (Dots)', type: 'dots' },
        { chars: '≡ ≡', size: 'N/A', label: 'Binocular Balance', type: 'bino' }
    ],
    4: [ // Numbers
        { chars: '8', size: '6/60', label: 'Numbers 60' },
        { chars: '3 6', size: '6/36', label: 'Numbers 36' },
        { chars: '5 7 2', size: '6/24', label: 'Numbers 24' },
        { chars: '9 4 8 3', size: '6/18', label: 'Numbers 18' },
        { chars: '6 2 5 8 4', size: '6/12', label: 'Numbers 12' },
        { chars: '7 3 9 5 2 8', size: '6/9', label: 'Numbers 9' },
    ],
    5: [ // Landolt C / Others
        { chars: 'C', size: '6/60', label: 'Landolt C 60', type: 'landolt' },
        { chars: 'C C', size: '6/36', label: 'Landolt C 36', type: 'landolt' },
        { chars: 'C C C', size: '6/24', label: 'Landolt C 24', type: 'landolt' },
        { chars: 'C C C C', size: '6/18', label: 'Landolt C 18', type: 'landolt' },
        { chars: '⬜ ⬛', size: '6/6', label: 'Contrast', type: 'contrast' },
    ]
};

// Font size map for preview based on Snellen size
function getPreviewFontSize(size) {
    const map = {
        '6/60': '5rem', '6/48': '4.2rem', '6/36': '3.4rem', '6/30': '2.8rem',
        '6/24': '2.4rem', '6/18': '1.85rem', '6/12': '1.4rem', '6/9': '1.1rem',
        '6/7.5': '0.95rem', '6/6': '0.8rem', '6/5': '0.7rem', '6/4': '0.6rem'
    };
    return map[size] || '1.2rem';
}

// ── Initialization ─────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateFooterTime();
    setInterval(updateFooterTime, 1000);
    updateCompareTable();
    updateRefractionTable();
    loadSubjectiveToControls();

    // Bind header input changes
    ['branchCode', 'staffName', 'patientName', 'patientAge', 'engagementId', 'customerId'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', () => {
                if (id === 'branchCode') session.branch.code = el.value;
                else if (id === 'staffName') session.branch.staff = el.value;
                else if (id === 'patientName') session.patient.name = el.value;
                else if (id === 'patientAge') session.patient.age = el.value;
                else if (id === 'engagementId') session.patient.engagementId = el.value;
                else if (id === 'customerId') session.patient.customerId = el.value;
            });
        }
    });

    // Bind AR and Lenso input changes
    bindDataInputs('ar', ['sph', 'cyl', 'axis', 'add', 'pd']);
    bindDataInputs('lenso', ['sph', 'cyl', 'axis', 'add']);

    // Bind subjective direct edits
    document.querySelectorAll('.param-value').forEach(input => {
        input.addEventListener('change', () => {
            const param = input.dataset.param;
            const val = input.value;
            if (session.activeEye === 'BOTH') {
                session.subjective.R[param] = val;
                session.subjective.L[param] = val;
            } else {
                session.subjective[session.activeEye][param] = val;
            }
            updateRefractionTable();
            updateCompareTable();
        });
    });

    // Initialize charts
    renderChartGrid(session.activeChartTab);

    // Bind Topcon Click Interaction
    bindTableInteractions();

    // Phase dropdown
    const phaseSelect = document.getElementById('phaseSelect');
    if (phaseSelect) {
        phaseSelect.value = session.currentPhase;
        phaseSelect.addEventListener('change', () => {
            setPhase(phaseSelect.value);
        });
    }
    updatePhaseUI();
    updateHeaderPd();
});

// ── Phase Configurations ──────────────────────────
const phaseConfigurations = {
    'distance_vision': {
        eyeMode: 'BOTH',
        chartTab: 2, // E-Chart
        chartIndex: 0, // E-Chart 400
        showJcc: false,
        showPinhole: true,
        occlude: null
    },
    'right_eye_refraction': {
        eyeMode: 'R',
        chartTab: 1, // Snellen Standard
        chartIndex: 0, // 200/150
        showJcc: false,
        occlude: 'L'
    },
    'jcc_axis_right': {
        eyeMode: 'R',
        chartTab: 3, // Specialty
        chartIndex: 1, // JCC Dots
        showJcc: true,
        occlude: 'L'
    },
    'jcc_power_right': {
        eyeMode: 'R',
        chartTab: 3, // Specialty
        chartIndex: 1, // JCC Dots
        showJcc: true,
        occlude: 'L'
    },
    'jcc_axis_left': {
        eyeMode: 'L',
        chartTab: 3, // Specialty
        chartIndex: 1, // JCC Dots
        showJcc: true,
        occlude: 'R'
    },
    'jcc_power_left': {
        eyeMode: 'L',
        chartTab: 3, // Specialty
        chartIndex: 1, // JCC Dots
        showJcc: true,
        occlude: 'R'
    },
    'duochrome_right': {
        eyeMode: 'R',
        chartTab: 3, // Specialty
        chartIndex: 0, // Duochrome
        showJcc: false,
        occlude: 'L'
    },
    'duochrome_left': {
        eyeMode: 'L',
        chartTab: 3, // Specialty
        chartIndex: 0, // Duochrome
        showJcc: false,
        occlude: 'R'
    },
    'left_eye_refraction': {
        eyeMode: 'L',
        chartTab: 1, // Snellen Standard
        chartIndex: 0, // 200/150
        showJcc: false,
        occlude: 'R'
    },
    'binocular_balance': {
        eyeMode: 'BOTH',
        chartTab: 3, // Specialty
        chartIndex: 2, // Bino Lines
        showJcc: false,
        occlude: null
    }
};

function handlePhaseChange() {
    const phaseSelect = document.getElementById('phaseSelect');
    const phaseKey = phaseSelect.value;
    const config = phaseConfigurations[phaseKey];

    if (config) {
        // 1. Set Eye Mode
        setActiveEye(config.eyeMode);

        // 2. Set Chart Tab & Item
        selectChartTab(config.chartTab);
        if (config.chartIndex !== undefined) {
            selectChartItem(config.chartIndex);
        }

        // 3. Toggle JCC Visibility (Integrated window within refraction card)
        const jccWindow = document.getElementById('jccWindow');
        if (jccWindow) {
            jccWindow.classList.toggle('hidden', !config.showJcc);
        }

        // 4. Toggle Pinhole Visibility (Distance Vision Only)
        const pinholeWindow = document.getElementById('pinholeWindow');
        if (pinholeWindow) {
            pinholeWindow.classList.toggle('hidden', !config.showPinhole);
        }
        // Auto-reset pinhole when leaving Distance Vision
        if (!config.showPinhole && session.isPinholeActive) {
            session.isPinholeActive = false;
            document.getElementById('btn-pinhole')?.classList.remove('active');
        }

        // 5. Set JCC Mode based on phase
        if (phaseKey.includes('jcc_axis')) setJCCMode('axis');
        if (phaseKey.includes('jcc_power')) setJCCMode('power');

        // 6. Update Restricted View (Occlusion)
        updateOcclusion(config.occlude);

        addLogEntry(`Phase changed to ${phaseKey}`, 'phase');
    }
}

function togglePinhole() {
    if (session.finalized) return;

    session.isPinholeActive = !session.isPinholeActive;
    const btn = document.getElementById('btn-pinhole');
    if (btn) {
        btn.classList.toggle('active', session.isPinholeActive);
        const icon = btn.querySelector('i');
        if (icon) {
            icon.className = session.isPinholeActive ? 'ph-bold ph-eye' : 'ph-bold ph-eye-closed';
        }
    }

    addLogEntry(`Pinhole Lens: ${session.isPinholeActive ? 'Engaged' : 'Disengaged'}`, 'special');
}

function updateOcclusion(occludedEye) {
    const table = document.getElementById('refractionTable');
    if (!table) return;

    // Reset occlusions
    table.querySelectorAll('.rt-val').forEach(el => el.classList.remove('occluded'));
    table.querySelectorAll('th').forEach(el => el.classList.remove('occluded-header'));

    if (occludedEye) {
        table.querySelectorAll(`[data-eye="${occludedEye}"]`).forEach(el => {
            el.classList.add('occluded');
        });
        const headerId = occludedEye === 'R' ? 'rt-eye-r-header' : 'rt-eye-l-header';
        const header = document.getElementById(headerId);
        if (header) header.classList.add('occluded-header');
    }
}

// ── Interaction ──────────────────────────────────
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

function handleTableMousedown(event, el) {
    if (session.finalized) {
        showToast('Session finalized — unlock to edit', 'warning');
        return;
    }

    // Check if cell is occluded
    if (el.classList.contains('occluded')) {
        showToast('This eye is currently occluded in this phase', 'info');
        return;
    }

    // button 0 = left, button 2 = right
    const action = (event.button === 0) ? 'subtract' : (event.button === 2 ? 'add' : null);
    if (!action) return;

    // Block default behavior (like context menu) instantly
    event.preventDefault();
    event.stopPropagation();

    const param = el.dataset.param;
    const eye = el.dataset.eye; // Could be R, L, or undefined (for Label/Center Column)

    // Determine precise delta (Axis now 5 as requested by user in previous turn)
    let delta = 0.25;
    if (param === 'axis') delta = 5;
    if (param === 'pd') delta = 0.5;

    // Left click subtracts, Right click adds
    if (action === 'subtract') delta = -delta;

    // Apply adjustment
    if (param === 'pd') {
        adjustValue('pd', delta);
    } else if (!eye) {
        // CENTER COLUMN CLICK: Adjust BOTH eyes
        const originalActive = session.activeEye;
        session.activeEye = 'BOTH';
        adjustValue(param, delta);
        session.activeEye = originalActive;
    } else {
        // Eye-specific click
        if (session.activeEye === 'BOTH') {
            adjustValue(param, delta);
        } else {
            // Force adjustment for ONLY the clicked eye (already set by phase, but fallback)
            const originalActive = session.activeEye;
            session.activeEye = eye;
            adjustValue(param, delta);
            session.activeEye = originalActive;
        }
    }
}

function bindDataInputs(type, fields) {
    ['r', 'l'].forEach(eye => {
        fields.forEach(field => {
            const el = document.getElementById(`${type}-${eye}-${field}`);
            if (el) {
                el.addEventListener('change', () => {
                    session[type][eye.toUpperCase()][field] = el.value;
                    updateCompareTable();
                });
            }
        });
    });
}

// ── JCC Logic ──────────────────────────────────────
function setJCCFlip(flip) {
    if (session.finalized) return;
    session.jccActiveFlip = flip;

    // Update buttons
    document.querySelectorAll('.jcc-flip-btn').forEach(btn => {
        btn.classList.toggle('active', btn.id === `btn-flip${flip}`);
    });

    // Update session log
    addLogEntry(`JCC: Switched to Flip ${flip}`, 'jcc');

    // Update dial rotation/indicator
    const dial = document.getElementById('jccDiamond');
    if (dial) {
        dial.classList.toggle('flipped', flip === 2);
    }
}

function setJCCMode(mode) {
    if (session.finalized) return;
    session.jccMode = mode;

    // Update manual toggle buttons in JCC window
    document.getElementById('btn-jcc-axis')?.classList.toggle('active', mode === 'axis');
    document.getElementById('btn-jcc-power')?.classList.toggle('active', mode === 'power');

    // Default to Flip 1 when switching modes
    setJCCFlip(1);

    addLogEntry(`JCC Mode: ${mode.toUpperCase()}`, 'jcc');
}

// ── Session ────────────────────────────────────────
function finalizeSession() {
    session.finalized = true;
    document.body.classList.add('session-finalized');
    showToast('Session finalized and locked', 'success');
}

function resetSession() {
    if (confirm('Reset all values?')) {
        location.reload();
    }
}

// ── Card Collapse ──────────────────────────────────
function toggleCard(cardId) {
    const card = document.getElementById(cardId);
    if (card) card.classList.toggle('collapsed');
}

// ── Active Eye ─────────────────────────────────────
function setActiveEye(eye) {
    session.activeEye = eye;
    document.querySelectorAll('.eye-toggle-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.eye === eye);
    });
    // loadSubjectiveToControls(); // Removed as controls are gone
}

// ── Adjust Value ───────────────────────────────────
function adjustValue(param, delta) {
    if (session.finalized) {
        showToast('Session finalized — unlock to edit', 'warning');
        return;
    }

    const eyes = session.activeEye === 'BOTH' ? ['R', 'L'] : [session.activeEye];

    eyes.forEach(eye => {
        let current = parseFloat(session.subjective[eye][param]) || 0;

        // If adjusting PD in BOTH mode, we split the total delta between eyes
        // To ensure precision, we don't round until display
        let actualDelta = delta;
        if (param === 'pd' && session.activeEye === 'BOTH') {
            actualDelta = delta / 2;
        }

        let newVal = current + actualDelta;

        // Axis wrapping 1-180
        if (param === 'axis') {
            newVal = (newVal % 180);
            if (newVal <= 0) newVal += 180;
        }

        // Save to state
        if (param === 'axis') {
            session.subjective[eye][param] = String(Math.round(newVal));
        } else if (param === 'pd') {
            // Store as string but keep more precision internally if needed
            // For PD, 32.0 - 0.25 = 31.75. 
            // We'll store it such that sum is correct.
            session.subjective[eye][param] = String(newVal);
        } else {
            session.subjective[eye][param] = formatDiopter(newVal);
        }
    });

    // Update visibility
    updateRefractionTable();
    updateCompareTable();

    const newValDisp = session.activeEye === 'BOTH' ? session.subjective.R[param] : session.subjective[session.activeEye][param];
    addLogEntry(`${param.toUpperCase()} ${delta >= 0 ? '+' : ''}${param === 'axis' ? delta : (param === 'pd' ? delta : formatDiopter(delta))} → ${newValDisp} [${session.activeEye}]`, 'adjust');
}

function formatDiopter(val) {
    const num = parseFloat(val);
    if (isNaN(num)) return '0.00';
    return (num >= 0 ? '+' : '') + num.toFixed(2);
}

// ── Refraction Table (TopCon-style R-center-L) ─────
function updateRefractionTable() {
    ['R', 'L'].forEach(eye => {
        const e = eye.toLowerCase();
        ['sph', 'cyl', 'axis', 'add'].forEach(param => {
            const el = document.getElementById(`rt-${e}-${param}`);
            if (el) el.textContent = session.subjective[eye][param] || '—';
        });
    });

    // PD shows summed value (Precision fixed by rounding the SUM, not the parts)
    const pdR = parseFloat(session.subjective.R.pd) || 32.0;
    const pdL = parseFloat(session.subjective.L.pd) || 32.0;
    const pdEl = document.getElementById('rt-pd');
    if (pdEl) pdEl.textContent = (pdR + pdL).toFixed(1);
}

// ── Phase ───────────────────────────────────────────
function setPhase(phaseId) {
    if (!REFRACTION_PHASES.some(p => p.id === phaseId)) return;
    session.currentPhase = phaseId;
    const phaseSelect = document.getElementById('phaseSelect');
    if (phaseSelect) phaseSelect.value = phaseId;
    updatePhaseUI();
    addLogEntry(`Phase: ${REFRACTION_PHASES.find(p => p.id === phaseId).label}`, 'info');
}

function updatePhaseUI() {
    const isJcc = JCC_PHASE_IDS.includes(session.currentPhase);
    const jccPanel = document.getElementById('refractionJccPanel');
    if (jccPanel) {
        jccPanel.hidden = !isJcc;
    }
    const phaseSelect = document.getElementById('phaseSelect');
    if (phaseSelect) phaseSelect.value = session.currentPhase;
}

// ── Header PD (moved from refraction table) ─────────
function updateHeaderPd() {
    const pdR = parseFloat(session.subjective.R.pd) || 32.0;
    const pdL = parseFloat(session.subjective.L.pd) || 32.0;
    const total = pdR + pdL;
    const el = document.getElementById('headerPdValue');
    if (el) el.textContent = total.toFixed(1);
}

function adjustHeaderPd(delta) {
    if (session.finalized) {
        showToast('Session finalized — unlock to edit', 'warning');
        return;
    }
    // Delta is change in total PD; split across eyes when BOTH
    const perEyeDelta = session.activeEye === 'BOTH' ? delta / 2 : delta;
    const eyes = session.activeEye === 'BOTH' ? ['R', 'L'] : [session.activeEye];
    eyes.forEach(eye => {
        let current = parseFloat(session.subjective[eye].pd) || 32.0;
        let newVal = Math.max(0, current + perEyeDelta);
        session.subjective[eye].pd = newVal.toFixed(1);
    });
    updateHeaderPd();
    const total = (parseFloat(session.subjective.R.pd) + parseFloat(session.subjective.L.pd)).toFixed(1);
    addLogEntry(`PD ${delta >= 0 ? '+' : ''}${delta} → ${total}`, 'adjust');
}

// ── Compare Table ──────────────────────────────────
function updateCompareTable() {
    const body = document.getElementById('compareBody');
    if (!body) return;

    const params = ['sph', 'cyl', 'axis', 'add'];
    const rows = [];

    ['R', 'L'].forEach(eye => {
        rows.push(`<tr><td colspan="4" style="padding:4px 4px 1px;"><span class="eye-badge ${eye === 'R' ? 'right' : 'left'}">${eye}</span></td></tr>`);
        params.forEach(p => {
            const ar = session.ar[eye][p] || '—';
            const lenso = session.lenso[eye][p] || '—';
            const subj = session.subjective[eye][p] || '—';
            rows.push(`
                <tr>
                    <td class="compare-label">${p.toUpperCase()}</td>
                    <td>${ar}</td>
                    <td>${lenso}</td>
                    <td>${subj}</td>
                </tr>
            `);
        });
    });

    body.innerHTML = rows.join('');
}

// ══════════════════════════════════════════════════
// CHART SELECTOR
// ══════════════════════════════════════════════════

function selectChartTab(tabNum) {
    session.activeChartTab = tabNum;
    session.activeChartIndex = 0;

    // Update tab UI
    document.querySelectorAll('.chart-tab').forEach(tab => {
        tab.classList.toggle('active', parseInt(tab.dataset.chart) === tabNum);
    });

    renderChartGrid(tabNum);
    addLogEntry(`Switched to Chart ${tabNum}`, 'chart');
}

function renderChartGrid(tabNum) {
    const grid = document.getElementById('chartGrid');
    const charts = chartData[tabNum] || [];
    grid.innerHTML = '';

    charts.forEach((chart, index) => {
        const thumb = document.createElement('div');
        thumb.className = 'chart-thumb' + (index === session.activeChartIndex ? ' active' : '');
        thumb.onclick = () => selectChartItem(index);

        // Optotype preview
        const optEl = document.createElement('div');
        optEl.className = 'chart-thumb-optotype';

        if (chart.type === 'e-chart') {
            // Random rotations for E characters
            const rotations = [0, 90, 180, 270];
            const eCount = chart.chars.split(' ').length;
            let eHtml = '';
            for (let i = 0; i < Math.min(eCount, 3); i++) {
                const rot = rotations[Math.floor(Math.random() * rotations.length)];
                eHtml += `<span style="display:inline-block;transform:rotate(${rot}deg);margin:0 1px">E</span>`;
            }
            optEl.innerHTML = eHtml;
        } else if (chart.type === 'landolt') {
            const rotations = [0, 45, 90, 135, 180, 225, 270, 315];
            const cCount = chart.chars.split(' ').length;
            let cHtml = '';
            for (let i = 0; i < Math.min(cCount, 3); i++) {
                const rot = rotations[Math.floor(Math.random() * rotations.length)];
                cHtml += `<span style="display:inline-block;transform:rotate(${rot}deg);margin:0 1px">C</span>`;
            }
            optEl.innerHTML = cHtml;
        } else {
            // Just show the first character(s)
            const displayChar = chart.chars.length > 5 ? chart.chars.substring(0, 5) : chart.chars;
            optEl.textContent = displayChar;
        }

        // Size label
        const sizeEl = document.createElement('div');
        sizeEl.className = 'chart-thumb-size';
        sizeEl.textContent = chart.size;

        thumb.appendChild(optEl);
        thumb.appendChild(sizeEl);
        grid.appendChild(thumb);
    });

    // Update preview
    if (charts.length > 0) {
        updateChartPreview(charts[session.activeChartIndex]);
    }
}

function selectChartItem(index) {
    const charts = chartData[session.activeChartTab] || [];
    if (index < 0 || index >= charts.length) return;

    session.activeChartIndex = index;

    // Update active thumb
    document.querySelectorAll('.chart-thumb').forEach((thumb, i) => {
        thumb.classList.toggle('active', i === index);
    });

    updateChartPreview(charts[index]);
    addLogEntry(`Selected ${charts[index].label} (${charts[index].size})`, 'chart');
}

function updateChartPreview(chart) {
    const previewEl = document.getElementById('chartPreviewOptotype');
    const sizeLabel = document.getElementById('chartSizeLabel');

    if (!previewEl || !chart) return;

    const fontSize = getPreviewFontSize(chart.size);

    if (chart.type === 'e-chart') {
        // Render tumbling E's
        const rotations = [0, 90, 180, 270];
        const eCount = chart.chars.split(' ').length;
        let html = '<div style="display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;">';
        for (let i = 0; i < eCount; i++) {
            const rot = rotations[Math.floor(Math.random() * rotations.length)];
            html += `<span style="font-size:${fontSize};display:inline-block;transform:rotate(${rot}deg)">E</span>`;
        }
        html += '</div>';
        previewEl.innerHTML = html;
    } else if (chart.type === 'landolt') {
        const rotations = [0, 45, 90, 135, 180, 225, 270, 315];
        const cCount = chart.chars.split(' ').length;
        let html = '<div style="display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;">';
        for (let i = 0; i < cCount; i++) {
            const rot = rotations[Math.floor(Math.random() * rotations.length)];
            html += `<span style="font-size:${fontSize};display:inline-block;transform:rotate(${rot}deg)">C</span>`;
        }
        html += '</div>';
        previewEl.innerHTML = html;
    } else {
        previewEl.innerHTML = `<span style="font-size:${fontSize};letter-spacing:4px">${chart.chars}</span>`;
    }

    if (sizeLabel) sizeLabel.textContent = chart.size;
}

// ══════════════════════════════════════════════════
// JCC — Jackson Cross Cylinder
// ══════════════════════════════════════════════════

function setJccMode(mode) {
    session.jccMode = mode;
    session.jccActiveFlip = null;

    // Update mode toggle UI
    document.querySelectorAll('.jcc-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Update diamond and label
    const diamond = document.getElementById('jccDiamond');
    const label = document.getElementById('jccModeLabel');
    if (diamond) {
        diamond.classList.toggle('flipped', mode === 'power');
    }
    if (label) {
        label.textContent = mode === 'axis' ? 'AXIS' : 'POWER';
    }

    // Reset flip buttons
    document.getElementById('jccFlip1Btn').classList.remove('active');
    document.getElementById('jccFlip2Btn').classList.remove('active');

    addLogEntry(`JCC mode: ${mode === 'axis' ? 'Axis Refine' : 'Power Refine'}`, 'jcc');
}

function jccFlip(flipNum) {
    session.jccActiveFlip = flipNum;

    // Update flip button UI
    document.getElementById('jccFlip1Btn').classList.toggle('active', flipNum === 1);
    document.getElementById('jccFlip2Btn').classList.toggle('active', flipNum === 2);

    addLogEntry(`JCC Flip ${flipNum} shown [${session.jccMode}]`, 'jcc');
}

function jccVerdict(verdict) {
    const modeLabel = session.jccMode === 'axis' ? 'Axis' : 'Power';
    let message = '';

    if (verdict === 'flip1') {
        message = `JCC ${modeLabel}: Flip 1 Better`;
    } else if (verdict === 'flip2') {
        message = `JCC ${modeLabel}: Flip 2 Better`;
    } else {
        message = `JCC ${modeLabel}: Same (No Change)`;
    }

    showToast(message, 'info');
    addLogEntry(message + ` [${session.activeEye}]`, 'jcc');

    // Reset flips
    session.jccActiveFlip = null;
    document.getElementById('btn-flip1')?.classList.remove('active');
    document.getElementById('btn-flip2')?.classList.remove('active');
}

// ── Save / Load AR & Lenso ─────────────────────────
function saveData(type) {
    const key = `phoropter_${type}`;
    const data = {};

    ['R', 'L'].forEach(eye => {
        data[eye] = {};
        const fields = type === 'ar' ? ['sph', 'cyl', 'axis', 'add', 'pd'] : ['sph', 'cyl', 'axis', 'add'];
        fields.forEach(f => {
            const el = document.getElementById(`${type}-${eye.toLowerCase()}-${f}`);
            data[eye][f] = el ? el.value : '';
            session[type][eye][f] = data[eye][f];
        });
    });

    localStorage.setItem(key, JSON.stringify(data));
    showToast(`${type.toUpperCase()} power saved`, 'success');
    addLogEntry(`${type.toUpperCase()} power saved to storage`, 'save');
    updateCompareTable();
}

function loadData(type) {
    const key = `phoropter_${type}`;
    const raw = localStorage.getItem(key);
    if (!raw) {
        showToast(`No saved ${type.toUpperCase()} data found`, 'warning');
        return;
    }

    try {
        const data = JSON.parse(raw);
        ['R', 'L'].forEach(eye => {
            if (!data[eye]) return;
            Object.keys(data[eye]).forEach(f => {
                const el = document.getElementById(`${type}-${eye.toLowerCase()}-${f}`);
                if (el) el.value = data[eye][f];
                session[type][eye][f] = data[eye][f];
            });
        });
        showToast(`${type.toUpperCase()} power loaded`, 'success');
        addLogEntry(`${type.toUpperCase()} power loaded from storage`, 'save');
        updateCompareTable();
    } catch (e) {
        showToast('Error loading data', 'error');
    }
}

// ── Memory Slots ───────────────────────────────────
function saveToSlot(index) {
    const snapshot = {
        R: { ...session.subjective.R },
        L: { ...session.subjective.L },
        timestamp: new Date().toLocaleTimeString()
    };
    session.memorySlots[index] = snapshot;

    const slot = document.getElementById(`memSlot${index}`);
    slot.classList.add('filled');
    document.getElementById(`slotStatus${index}`).textContent = snapshot.timestamp;
    document.getElementById(`slotPreview${index}`).innerHTML =
        `R: S${snapshot.R.sph} C${snapshot.R.cyl} A${snapshot.R.axis}<br>` +
        `L: S${snapshot.L.sph} C${snapshot.L.cyl} A${snapshot.L.axis}`;
    document.getElementById(`recallBtn${index}`).disabled = false;
    document.getElementById(`clearBtn${index}`).disabled = false;

    showToast(`Saved to M${index + 1}`, 'success');
    addLogEntry(`Saved to Memory M${index + 1}`, 'memory');
}

function recallSlot(index) {
    const snapshot = session.memorySlots[index];
    if (!snapshot) return;

    session.subjective.R = { ...snapshot.R };
    session.subjective.L = { ...snapshot.L };
    loadSubjectiveToControls();
    updateRefractionTable();
    updateCompareTable();

    showToast(`Recalled M${index + 1}`, 'info');
    addLogEntry(`Recalled Memory M${index + 1}`, 'memory');
}

function clearSlot(index) {
    session.memorySlots[index] = null;
    const slot = document.getElementById(`memSlot${index}`);
    slot.classList.remove('filled');
    document.getElementById(`slotStatus${index}`).textContent = 'Empty';
    document.getElementById(`slotPreview${index}`).textContent = '—';
    document.getElementById(`recallBtn${index}`).disabled = true;
    document.getElementById(`clearBtn${index}`).disabled = true;
}

// ── Export ──────────────────────────────────────────
function exportCSV() {
    readHeaderFields();
    const lines = [];

    lines.push('Section,Eye,SPH,CYL,AXIS,ADD,PD');
    lines.push(`AR,R,${v(session.ar.R.sph)},${v(session.ar.R.cyl)},${v(session.ar.R.axis)},${v(session.ar.R.add)},${v(session.ar.R.pd)}`);
    lines.push(`AR,L,${v(session.ar.L.sph)},${v(session.ar.L.cyl)},${v(session.ar.L.axis)},${v(session.ar.L.add)},${v(session.ar.L.pd)}`);
    lines.push(`Lenso,R,${v(session.lenso.R.sph)},${v(session.lenso.R.cyl)},${v(session.lenso.R.axis)},${v(session.lenso.R.add)},`);
    lines.push(`Lenso,L,${v(session.lenso.L.sph)},${v(session.lenso.L.cyl)},${v(session.lenso.L.axis)},${v(session.lenso.L.add)},`);
    lines.push(`Subjective,R,${v(session.subjective.R.sph)},${v(session.subjective.R.cyl)},${v(session.subjective.R.axis)},${v(session.subjective.R.add)},${v(session.subjective.R.pd)}`);
    lines.push(`Subjective,L,${v(session.subjective.L.sph)},${v(session.subjective.L.cyl)},${v(session.subjective.L.axis)},${v(session.subjective.L.add)},${v(session.subjective.L.pd)}`);

    session.memorySlots.forEach((slot, i) => {
        if (slot) {
            lines.push(`Memory${i + 1},R,${v(slot.R.sph)},${v(slot.R.cyl)},${v(slot.R.axis)},${v(slot.R.add)},${v(slot.R.pd)}`);
            lines.push(`Memory${i + 1},L,${v(slot.L.sph)},${v(slot.L.cyl)},${v(slot.L.axis)},${v(slot.L.add)},${v(slot.L.pd)}`);
        }
    });

    lines.push('');
    lines.push('Patient Info');
    lines.push(`Name,${session.patient.name}`);
    lines.push(`Age,${session.patient.age}`);
    lines.push(`EngagementID,${session.patient.engagementId}`);
    lines.push(`CustomerID,${session.patient.customerId}`);
    lines.push(`Branch,${session.branch.code}`);
    lines.push(`Staff,${session.branch.staff}`);

    downloadFile(`prescription_${session.patient.engagementId || 'session'}.csv`, lines.join('\n'), 'text/csv');
    showToast('CSV exported', 'success');
    addLogEntry('Exported CSV', 'export');
}

function exportJSON() {
    readHeaderFields();
    const data = {
        patient: session.patient,
        branch: session.branch,
        ar: session.ar,
        lenso: session.lenso,
        subjective: session.subjective,
        memorySlots: session.memorySlots,
        timestamp: new Date().toISOString(),
        finalized: session.finalized
    };

    downloadFile(
        `prescription_${session.patient.engagementId || 'session'}.json`,
        JSON.stringify(data, null, 2),
        'application/json'
    );
    showToast('JSON exported', 'success');
    addLogEntry('Exported JSON', 'export');
}

function v(val) { return val || ''; }

function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function readHeaderFields() {
    session.patient.name = document.getElementById('patientName').value;
    session.patient.age = document.getElementById('patientAge').value;
    session.patient.engagementId = document.getElementById('engagementId').value;
    session.patient.customerId = document.getElementById('customerId').value;
    session.branch.code = document.getElementById('branchCode').value;
    session.branch.staff = document.getElementById('staffName').value;
}

// ── Finalize / New Session / Reset ─────────────────
function finalizePrescription() {
    if (session.finalized) {
        session.finalized = false;
        document.getElementById('sessionStatus').querySelector('span:last-child').textContent = 'Ready';
        document.getElementById('sessionStatus').querySelector('.status-dot').style.background = 'var(--accent-green)';
        showToast('Session unlocked', 'info');
        addLogEntry('Session unlocked', 'save');
        return;
    }

    readHeaderFields();
    session.finalized = true;
    document.getElementById('sessionStatus').querySelector('span:last-child').textContent = 'Finalized';
    document.getElementById('sessionStatus').querySelector('.status-dot').style.background = 'var(--accent-amber)';
    showToast('Prescription finalized! Export now.', 'success');
    addLogEntry('Prescription finalized', 'save');
}

function newSession() {
    if (!confirm('Start a new session? Unsaved data will be lost.')) return;
    resetAll();
    showToast('New session started', 'info');
}

function resetAll() {
    session.patient = { name: '', age: '', engagementId: '', customerId: '' };
    session.branch = { code: '', staff: '' };
    session.ar = {
        R: { sph: '', cyl: '', axis: '', add: '', pd: '' },
        L: { sph: '', cyl: '', axis: '', add: '', pd: '' }
    };
    session.lenso = {
        R: { sph: '', cyl: '', axis: '', add: '' },
        L: { sph: '', cyl: '', axis: '', add: '' }
    };
    session.subjective = {
        R: { sph: '0.00', cyl: '0.00', axis: '180', add: '0.00', pd: '32.0' },
        L: { sph: '0.00', cyl: '0.00', axis: '180', add: '0.00', pd: '32.0' }
    };
    session.memorySlots = [null, null, null];
    session.log = [];
    session.finalized = false;

    ['branchCode', 'staffName', 'patientName', 'patientAge', 'engagementId', 'customerId'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });

    document.querySelectorAll('.cell-input').forEach(el => el.value = '');

    loadSubjectiveToControls();
    setActiveEye('R');
    updateRefractionTable();
    updateCompareTable();

    [0, 1, 2].forEach(clearSlot);
    clearLog();

    session.currentPhase = 'distance_vision';
    const phaseSelect = document.getElementById('phaseSelect');
    if (phaseSelect) phaseSelect.value = session.currentPhase;
    updatePhaseUI();

    // Reset chart
    selectChartTab(1);

    // Reset JCC
    setJccMode('axis');
    updateHeaderPd();

    document.getElementById('sessionStatus').querySelector('span:last-child').textContent = 'Ready';
    document.getElementById('sessionStatus').querySelector('.status-dot').style.background = 'var(--accent-green)';
}

// ── Video ──────────────────────────────────────────
function connectVideo() {
    const urlInput = document.getElementById('videoUrl');
    const video = document.getElementById('liveVideo');
    const placeholder = document.getElementById('videoPlaceholder');

    if (urlInput.value.trim()) {
        video.src = urlInput.value.trim();
        video.style.display = 'block';
        placeholder.style.display = 'none';
        showToast('Video stream connected', 'success');
    } else {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                    video.style.display = 'block';
                    placeholder.style.display = 'none';
                    showToast('Camera connected', 'success');
                })
                .catch(() => {
                    showToast('Camera access denied', 'error');
                });
        } else {
            showToast('No camera available', 'warning');
        }
    }
}

// ── Session Log ────────────────────────────────────
function addLogEntry(message, type = 'info') {
    const now = new Date().toLocaleTimeString('en-US', { hour12: false });
    const entry = { time: now, message, type };
    session.log.push(entry);

    const logEl = document.getElementById('sessionLog');
    const emptyMsg = logEl.querySelector('.log-empty');
    if (emptyMsg) emptyMsg.remove();

    const div = document.createElement('div');
    div.className = `log-entry log-${type}`;
    div.innerHTML = `<span class="log-time">${now}</span>${message}`;
    logEl.prepend(div);

    while (logEl.children.length > 100) {
        logEl.lastChild.remove();
    }
}

function clearLog() {
    session.log = [];
    const logEl = document.getElementById('sessionLog');
    logEl.innerHTML = '<div class="log-empty">No adjustments yet</div>';
}

// ── Toast ──────────────────────────────────────────
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}

// ── Footer Time ────────────────────────────────────
function updateFooterTime() {
    const el = document.getElementById('footerTime');
    if (el) el.textContent = new Date().toLocaleString('en-IN', {
        dateStyle: 'medium',
        timeStyle: 'short'
    });
}
