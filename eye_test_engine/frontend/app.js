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

let operatorName = '';  // cached optometrist name

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

// ── Device Management ────────────────────────────────

async function fetchDevices() {
    const select = document.getElementById('phoropterIdInput');
    if (!select) return;

    try {
        const resp = await fetch(`${CONFIG.backendUrl}/api/devices?all=true`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const devices = Array.isArray(data) ? data : (data.devices || []);
        select.innerHTML = '';

        if (devices.length === 0) {
            select.innerHTML = '<option value="">No devices found</option>';
            return;
        }

        const savedId = localStorage.getItem('phoropterId');

        devices.forEach(dev => {
            const id = dev.device_id || dev.id || dev.name || '';
            const status = dev.status || '';
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = `${id} (${status})`;
            if (id === savedId) opt.selected = true;
            select.appendChild(opt);
        });

        if (select.value) {
            localStorage.setItem('phoropterId', select.value);
        }
        onDeviceSelectionChanged();
    } catch (err) {
        console.warn('Could not fetch devices:', err);
        select.innerHTML = '<option value="phoropter-1">phoropter-1 (default)</option>';
        localStorage.setItem('phoropterId', 'phoropter-1');
    }
}

function onDeviceSelectionChanged() {
    const select = document.getElementById('phoropterIdInput');
    const acquireBtn = document.getElementById('acquireDeviceBtn');
    if (!select) return;

    const id = select.value;
    if (id) {
        localStorage.setItem('phoropterId', id);
        if (acquireBtn && !_deviceAcquired) acquireBtn.style.display = 'inline-block';
    } else {
        if (acquireBtn) acquireBtn.style.display = 'none';
    }
}

let _cachedClientIp = null;

async function getClientIp() {
    if (_cachedClientIp) return _cachedClientIp;
    try {
        const resp = await fetch('https://api.ipify.org?format=json');
        const data = await resp.json();
        _cachedClientIp = data.ip || 'unknown';
    } catch {
        _cachedClientIp = 'unknown';
    }
    return _cachedClientIp;
}

async function getBrainId() {
    const ip = await getClientIp();
    const name = (operatorName || 'unknown').replace(/\s+/g, '_');
    return `${name}@${ip}`;
}

async function acquireSelectedDevice() {
    const deviceId = CONFIG.phoropterId;
    if (!deviceId) return;

    const btn = document.getElementById('acquireDeviceBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Acquiring...'; }

    try {
        const brainId = await getBrainId();
        const resp = await fetch(`${CONFIG.backendUrl}/api/devices/${deviceId}/acquire`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ brain_id: brainId, name: operatorName || 'Eye Test UI' })
        });
        const data = await resp.json();

        if (resp.ok) {
            _deviceAcquired = true;
            if (btn) btn.style.display = 'none';
            document.getElementById('phoropterIdInput').disabled = true;
            console.log('Device acquired:', deviceId, data);
        } else {
            alert(`Could not acquire ${deviceId}: ${data.error || data.reason || resp.status}`);
            if (btn) { btn.disabled = false; btn.textContent = 'Acquire'; }
        }
    } catch (err) {
        alert(`Failed to acquire device: ${err.message}`);
        if (btn) { btn.disabled = false; btn.textContent = 'Acquire'; }
    }
}

async function releaseDevice() {
    const deviceId = CONFIG.phoropterId;
    if (!deviceId) return;

    try {
        const brainId = await getBrainId();
        await fetch(`${CONFIG.backendUrl}/api/devices/${deviceId}/release`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ brain_id: brainId })
        });
        console.log('Device released:', deviceId);
    } catch (err) {
        console.warn('Could not release device:', err);
    }

    _deviceAcquired = false;
    document.getElementById('phoropterIdInput').disabled = false;

    const btn = document.getElementById('acquireDeviceBtn');
    if (btn) {
        btn.style.display = 'inline-block';
        btn.disabled = false;
        btn.textContent = 'Acquire';
        btn.style.background = '';
    }
}

// Initialize
// ── Session Persistence (survives refresh) ───────────

const SESSION_STORAGE_KEY = 'eyeTestSession';

let _deviceAcquired = false;

function _saveSessionToStorage() {
    if (!sessionState.sessionId) return;
    const data = {
        sessionId: sessionState.sessionId,
        responseCount: sessionState.responseCount,
        storedPower: storedPower,
        currentAppliedPower: currentAppliedPower,
        deviceAcquired: _deviceAcquired,
        deviceId: CONFIG.phoropterId,
    };
    try { sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(data)); }
    catch (e) { console.warn('sessionStorage write failed:', e); }
}

function _clearSessionStorage() {
    try { sessionStorage.removeItem(SESSION_STORAGE_KEY); } catch (_) {}
}

async function _tryRestoreSession() {
    let saved;
    try { saved = JSON.parse(sessionStorage.getItem(SESSION_STORAGE_KEY)); }
    catch (_) { return false; }
    if (!saved || !saved.sessionId) return false;

    try {
        // 1. Verify internet / backend is reachable
        const statusResp = await fetch(`${CONFIG.backendUrl}/api/session/${saved.sessionId}/status`);
        if (!statusResp.ok) {
            console.warn('Backend session gone, starting fresh');
            _clearSessionStorage();
            return false;
        }

        // 2. If device was acquired, verify the lock is still alive via heartbeat
        if (saved.deviceAcquired && saved.deviceId) {
            const brainId = await getBrainId();
            const hbResp = await fetch(`${CONFIG.backendUrl}/api/devices/${saved.deviceId}/heartbeat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ brain_id: brainId })
            });
            if (hbResp.status !== 200 && hbResp.status !== 202) {
                console.warn(`Heartbeat returned ${hbResp.status}, device lock lost — starting fresh`);
                _clearSessionStorage();
                return false;
            }
        }

        const data = await statusResp.json();

        // Restore JS state
        sessionState.sessionId = saved.sessionId;
        sessionState.responseCount = saved.responseCount || data.total_rows || 0;
        storedPower = saved.storedPower || { ar: null, lenso: null };
        currentAppliedPower = saved.currentAppliedPower || 'none';

        // Restore UI
        document.getElementById('welcomeScreen').style.display = 'none';
        document.getElementById('testScreen').style.display = 'block';

        updateSessionInfo(data);
        displayQuestion(data);
        updateStatusIndicator(true);
        updatePowerButtonStates(currentAppliedPower);

        if (storedPower.ar) {
            document.getElementById('applyArBtn').disabled = false;
            document.getElementById('applyArBtn').title = 'Apply AR Power';
        }
        if (storedPower.lenso) {
            document.getElementById('applyLensoBtn').disabled = false;
            document.getElementById('applyLensoBtn').title = 'Apply Lenso Power';
        }

        // Restore device acquisition state
        if (saved.deviceAcquired && saved.deviceId) {
            _deviceAcquired = true;
            const select = document.getElementById('phoropterIdInput');
            if (select) { select.value = saved.deviceId; select.disabled = true; }
            const acqBtn = document.getElementById('acquireDeviceBtn');
            if (acqBtn) acqBtn.style.display = 'none';
        }

        addToHistory('Session restored after refresh', 'info');
        console.log('Session restored:', saved.sessionId);
        return true;
    } catch (err) {
        console.warn('No internet or backend unreachable, starting fresh:', err);
        _clearSessionStorage();
        return false;
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Eye Test Engine Frontend Loaded');

    updateStatusIndicator(false);
    populateDirectCommands();
    bindTableInteractions();
    checkOptometristName();
    fetchDevices();

    await _tryRestoreSession();
});

// ── Optometrist Name Cache (12-hour TTL) ─────────────

const OPTOMETRIST_CACHE_KEY = 'optometristName';
const OPTOMETRIST_TS_KEY = 'optometristNameTimestamp';
const OPTOMETRIST_TTL_MS = 12 * 60 * 60 * 1000;

function checkOptometristName() {
    const cached = localStorage.getItem(OPTOMETRIST_CACHE_KEY);
    const ts = parseInt(localStorage.getItem(OPTOMETRIST_TS_KEY) || '0', 10);
    const expired = (Date.now() - ts) > OPTOMETRIST_TTL_MS;

    if (cached && !expired) {
        operatorName = cached;
        return;
    }

    localStorage.removeItem(OPTOMETRIST_CACHE_KEY);
    localStorage.removeItem(OPTOMETRIST_TS_KEY);

    const modal = document.getElementById('optometristModal');
    if (modal) {
        modal.classList.add('active');
        const input = document.getElementById('optometristNameInput');
        if (input) setTimeout(() => input.focus(), 200);
    }
}

function saveOptometristName() {
    const input = document.getElementById('optometristNameInput');
    const name = (input ? input.value.trim() : '');
    if (!name) {
        input.style.borderColor = '#f44336';
        input.placeholder = 'Name is required';
        return;
    }
    operatorName = name;
    localStorage.setItem(OPTOMETRIST_CACHE_KEY, name);
    localStorage.setItem(OPTOMETRIST_TS_KEY, String(Date.now()));

    const modal = document.getElementById('optometristModal');
    if (modal) modal.classList.remove('active');
}

// ── Manual Refraction Adjustments ─────────────────────

let manualControlsLocked = false;
let _manualAutoUnlockTimer = null;
let typeModeActive = false;
let _typeModeEditing = false;

function _setManualLock(locked) {
    manualControlsLocked = locked;
    const btn = document.getElementById('manualLockBtn');
    const cells = document.querySelectorAll('.rt-val');
    if (locked) {
        if (btn) { btn.textContent = '🔒'; btn.classList.add('locked'); }
        cells.forEach(c => c.classList.add('locked'));
    } else {
        if (btn) { btn.textContent = '🔓'; btn.classList.remove('locked'); }
        cells.forEach(c => c.classList.remove('locked'));
    }
}

function toggleManualLock() {
    if (_manualAutoUnlockTimer) {
        clearTimeout(_manualAutoUnlockTimer);
        _manualAutoUnlockTimer = null;
    }
    if (!manualControlsLocked && typeModeActive) _exitTypeMode();
    _setManualLock(!manualControlsLocked);
}

// ── Type Mode ────────────────────────────────────────

function toggleTypeMode() {
    if (typeModeActive) {
        _exitTypeMode();
    } else {
        _enterTypeMode();
    }
}

function _enterTypeMode() {
    typeModeActive = true;
    if (manualControlsLocked) _setManualLock(false);

    const btn = document.getElementById('typeModeBtn');
    if (btn) btn.classList.add('active');

    document.querySelectorAll('.rt-val[data-param]').forEach(el => {
        el.classList.add('type-mode');
        el.setAttribute('data-tip', 'Click to type value');
    });
}

function _exitTypeMode() {
    _commitActiveInput(false);
    typeModeActive = false;
    _typeModeEditing = false;

    const btn = document.getElementById('typeModeBtn');
    if (btn) btn.classList.remove('active');

    document.querySelectorAll('.rt-val[data-param]').forEach(el => {
        el.classList.remove('type-mode');
        el.setAttribute('data-tip', 'Right-click = +  |  Left-click = −');
    });
}

function _getTypableFields() {
    return Array.from(document.querySelectorAll('.rt-val[data-param]'));
}

function _openInputInCell(cell) {
    if (!cell || cell.querySelector('.rt-type-input')) return;

    _typeModeEditing = true;
    const originalText = cell.textContent.trim();

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'rt-type-input';
    input.value = originalText.replace(/[+°]/g, '');
    input.dataset.originalValue = originalText;

    if (cell.dataset.param === 'axis') {
        input.inputMode = 'numeric';
        input.pattern = '[0-9]*';
    } else {
        input.inputMode = 'decimal';
    }

    input.addEventListener('keydown', (e) => _handleTypeInputKey(e, cell, input));
    input.addEventListener('blur', () => {
        setTimeout(() => {
            if (cell.contains(input)) {
                _restoreCell(cell, input.dataset.originalValue);
                _typeModeEditing = false;
            }
        }, 50);
    });

    cell.textContent = '';
    cell.appendChild(input);
    input.focus();
    input.select();
}

function _restoreCell(cell, text) {
    const input = cell.querySelector('.rt-type-input');
    if (input) input.remove();
    cell.textContent = text;
    _typeModeEditing = false;
}

function _handleTypeInputKey(e, cell, input) {
    if (e.key === 'Enter') {
        e.preventDefault();
        _commitActiveInput(true);
    } else if (e.key === 'Escape') {
        e.preventDefault();
        _restoreCell(cell, input.dataset.originalValue);
    } else if (e.key === 'Tab') {
        e.preventDefault();
        const parsed = _parseTypedValue(cell.dataset.param, input.value);
        if (parsed !== null) {
            _restoreCell(cell, _formatCellValue(cell.dataset.param, parsed));
        } else {
            _restoreCell(cell, input.dataset.originalValue);
        }
        const fields = _getTypableFields();
        const idx = fields.indexOf(cell);
        const next = fields[(idx + 1) % fields.length];
        _openInputInCell(next);
    }
}

function _parseTypedValue(param, raw) {
    const s = raw.replace(/[+°\s]/g, '').trim();
    if (s === '') return null;
    const n = parseFloat(s);
    if (isNaN(n)) return null;
    if (param === 'axis') {
        const rounded = Math.round(n / 5) * 5;
        if (rounded <= 0 || rounded > 180) return 180;
        return rounded;
    }
    if (param === 'cyl') {
        if (n > 0) return 0;
        return Math.round(n * 4) / 4;
    }
    if (param === 'add') {
        if (n < 0) return 0;
        return Math.round(n * 4) / 4;
    }
    return Math.round(n * 4) / 4;
}

function _formatCellValue(param, val) {
    if (param === 'axis') return String(Math.round(val));
    if (param === 'add') return '+' + val.toFixed(2);
    return (val >= 0 ? '+' : '') + val.toFixed(2);
}

async function _commitActiveInput(submit) {
    if (!submit) {
        document.querySelectorAll('.rt-type-input').forEach(input => {
            const cell = input.parentElement;
            if (cell) _restoreCell(cell, input.dataset.originalValue);
        });
        _typeModeEditing = false;
        return;
    }

    const fields = _getTypableFields();
    const power = {
        right: { sph: 0, cyl: 0, axis: 180, add: 0 },
        left:  { sph: 0, cyl: 0, axis: 180, add: 0 }
    };
    const oldPower = sessionState.lastResponse?.power || { right: {sph:0,cyl:0,axis:180,add:0}, left: {sph:0,cyl:0,axis:180,add:0} };
    let anyChanged = false;

    for (const cell of fields) {
        const eye = cell.dataset.eye === 'R' ? 'right' : 'left';
        const param = cell.dataset.param;
        const input = cell.querySelector('.rt-type-input');
        const rawText = input ? input.value : cell.textContent;
        const parsed = _parseTypedValue(param, rawText);

        if (parsed !== null) {
            power[eye][param] = parsed;
        } else {
            power[eye][param] = oldPower[eye]?.[param] || (param === 'axis' ? 180 : 0);
        }

        const orig = oldPower[eye]?.[param] || (param === 'axis' ? 180 : 0);
        if (Math.abs(power[eye][param] - orig) > 0.001) anyChanged = true;

        _restoreCell(cell, _formatCellValue(param, power[eye][param]));
    }

    _typeModeEditing = false;

    if (!anyChanged || !sessionState.sessionId) return;

    try {
        showLoading(true);
        const currentOccluder = document.getElementById('occluderState').textContent || 'BINO';

        if (sessionState.sessionId) {
            try {
                await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/sync-power`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(power)
                });
            } catch (syncErr) {
                console.warn('Failed to sync typed power to backend:', syncErr);
            }
        }

        await syncBrokerState(oldPower, currentOccluder);
        await setPower(power, currentOccluder);

        if (!sessionState.lastResponse) sessionState.lastResponse = {};
        sessionState.lastResponse.power = power;
        updateSessionInfo({ ...sessionState.lastResponse, power });

        sessionState.responseCount++;
        document.getElementById('responseCount').textContent = sessionState.responseCount;
        addToHistory('Typed power applied', 'adjust');
        _saveSessionToStorage();
    } catch (error) {
        console.error('Error applying typed power:', error);
        alert('Failed to apply typed power.');
    } finally {
        showLoading(false);
    }
}

function bindTableInteractions() {
    const table = document.getElementById('refractionTable');
    if (table) {
        table.oncontextmenu = function (e) {
            e.preventDefault();
            e.stopPropagation();
            return false;
        };
        table.querySelectorAll('.rt-val').forEach(el => {
            el.setAttribute('data-tip', 'R = +  |  L = −');
            el.addEventListener('mousedown', (event) => handleTableMousedown(event, el));
        });
    }
}

async function handleTableMousedown(event, el) {
    if (!sessionState.sessionId) {
        alert('Please start a test session first.');
        return;
    }

    if (typeModeActive) {
        event.preventDefault();
        event.stopPropagation();
        if (el.dataset.param) _openInputInCell(el);
        return;
    }

    if (manualControlsLocked) return;

    // button 0 = left, button 2 = right
    const action = (event.button === 0) ? 'subtract' : (event.button === 2 ? 'add' : null);
    if (!action) return;

    event.preventDefault();
    event.stopPropagation();

    const param = el.dataset.param;
    const eye = el.dataset.eye;

    let delta = 0.25;
    if (param === 'axis') delta = 5;

    if (action === 'subtract') delta = -delta;

    _setManualLock(true);
    if (_manualAutoUnlockTimer) clearTimeout(_manualAutoUnlockTimer);
    _manualAutoUnlockTimer = setTimeout(() => {
        _manualAutoUnlockTimer = null;
        _setManualLock(false);
    }, 1000);

    await applyManualPowerChange(eye, param, delta);
}

async function applyManualPowerChange(eye, param, delta) {
    if (!sessionState.lastResponse || !sessionState.lastResponse.power) return;

    const p = sessionState.lastResponse.power;
    const eyeKey = eye === 'R' ? 'right' : 'left';

    if (!p[eyeKey]) {
        p[eyeKey] = { sph: 0, cyl: 0, axis: 180 };
    }

    // Snapshot the pre-adjustment state for broker sync
    const prevPower = {
        right: { sph: p.right?.sph || 0, cyl: p.right?.cyl || 0, axis: p.right?.axis || 180 },
        left:  { sph: p.left?.sph  || 0, cyl: p.left?.cyl  || 0, axis: p.left?.axis  || 180 }
    };

    let current = parseFloat(p[eyeKey][param]) || 0;
    let newVal = current + delta;

    if (param === 'axis') {
        newVal = (Math.round(newVal) % 180);
        if (newVal <= 0) newVal += 180;
    }

    p[eyeKey][param] = newVal;

    // Update UI optimistically
    updateSessionInfo(sessionState.lastResponse);

    try {
        showLoading(true);
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

        // Tell the broker what the phoropter's actual state is before sending
        // the new target. JCC increase/decrease commands move the phoropter
        // without updating the broker's internal tracker, so without this the
        // broker would calculate clicks from a stale baseline.
        await syncBrokerState(prevPower, currentOccluder);
        await setPower(reqPower, currentOccluder);

        sessionState.responseCount++;
        document.getElementById('responseCount').textContent = sessionState.responseCount;
        addToHistory(`Manual Adjust: ${param.toUpperCase()} ${delta > 0 ? '+' : ''}${delta} [${eye}]`, 'adjust');
        _saveSessionToStorage();
    } catch (error) {
        console.error('Error applying manual power:', error);
        alert('Failed to push manual power to phoropter. Try again.');
    } finally {
        showLoading(false);
    }
}

async function syncBrokerState(power, occluder) {
    const right = power.right || { sph: 0, cyl: 0, axis: 180 };
    const left  = power.left  || { sph: 0, cyl: 0, axis: 180 };

    let auxLens = "OFF";
    if (occluder === "Left_Occluded") auxLens = "AuxLensL";
    else if (occluder === "Right_Occluded") auxLens = "AuxLensR";

    const rightEye = { sph: right.sph, cyl: right.cyl, axis: right.axis };
    const leftEye  = { sph: left.sph,  cyl: left.cyl,  axis: left.axis };
    if (right.add !== undefined && right.add !== 0) rightEye.add = right.add;
    if (left.add  !== undefined && left.add  !== 0) leftEye.add  = left.add;

    const payload = {
        right_eye: rightEye,
        left_eye:  leftEye,
        aux_lens: auxLens
    };

    try {
        await fetch(`${CONFIG.phoropterUrl}/phoropter/${CONFIG.phoropterId}/sync-state`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    } catch (err) {
        console.warn('Failed to sync broker state:', err);
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

        // Sync broker to current phoropter state before applying new absolute values
        const currentOccluder = document.getElementById('occluderState').textContent || 'BINO';
        if (sessionState.lastResponse && sessionState.lastResponse.power) {
            await syncBrokerState(sessionState.lastResponse.power, currentOccluder);
        }

        await setPower(power, 'BINO');

        currentAppliedPower = type;
        updatePowerButtonStates(type);

        sessionState.responseCount++;
        document.getElementById('responseCount').textContent = sessionState.responseCount;
        const label = type === 'ar' ? 'AR' : 'Lenso';
        addToHistory(`${label} power applied`, 'info');
        _saveSessionToStorage();

        // Update refraction table to reflect applied power
        const fmtSign = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);
        const rSph = document.getElementById('rt-r-sph');
        const rCyl = document.getElementById('rt-r-cyl');
        const rAxis = document.getElementById('rt-r-axis');
        const lSph = document.getElementById('rt-l-sph');
        const lCyl = document.getElementById('rt-l-cyl');
        const lAxis = document.getElementById('rt-l-axis');
        if (rSph) rSph.textContent = fmtSign(power.right.sph);
        if (rCyl) rCyl.textContent = fmtSign(power.right.cyl);
        if (rAxis) rAxis.textContent = power.right.axis.toFixed(0);
        if (lSph) lSph.textContent = fmtSign(power.left.sph);
        if (lCyl) lCyl.textContent = fmtSign(power.left.cyl);
        if (lAxis) lAxis.textContent = power.left.axis.toFixed(0);
        document.getElementById('occluderState').textContent = 'BINO';

        // Sync applied power to backend session state
        if (sessionState.sessionId) {
            try {
                await fetch(`${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/sync-power`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(power)
                });
            } catch (syncErr) {
                console.warn('Failed to sync applied power to backend:', syncErr);
            }
        }

        // Keep lastResponse in sync so manual adjustments and next QnA work from this baseline
        if (!sessionState.lastResponse) {
            sessionState.lastResponse = {};
        }
        sessionState.lastResponse.power = power;
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
        _saveSessionToStorage();

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
        _saveSessionToStorage();

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

    const rightEye = { sph: right.sph, cyl: right.cyl, axis: right.axis };
    const leftEye  = { sph: left.sph,  cyl: left.cyl,  axis: left.axis };
    if (right.add !== undefined && right.add !== 0) rightEye.add = right.add;
    if (left.add  !== undefined && left.add  !== 0) leftEye.add  = left.add;

    const payload = {
        test_cases: [{
            aux_lens: auxLens,
            right_eye: rightEye,
            left_eye: leftEye
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
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ar: storedPower.ar || null,
                lenso: storedPower.lenso || null,
                operator_name: operatorName || null
            })
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
    } finally {
        _clearSessionStorage();
        await releaseDevice();
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
    // Block intent shortcuts while type mode input is active
    if (_typeModeEditing) return;

    // Number keys 1-9 to select intents
    if (e.key >= '1' && e.key <= '9') {
        const index = parseInt(e.key) - 1;
        const buttons = document.querySelectorAll('.intent-button');
        if (buttons[index]) {
            buttons[index].click();
        }
    }
});
