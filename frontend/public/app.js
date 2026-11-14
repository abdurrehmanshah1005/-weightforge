import { initialState, applyEvent, chartData, baselineSummary, isBusy } from './lab-state.mjs';

const $ = id => document.getElementById(id);
const number = value => Number.isFinite(value) ? value.toFixed(4) : '—';
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
let state = initialState();
let sessionId = null;
let source = null;
let metric = 'loss';
let selected = null;
let reviewIndex = 0;
let connected = false;
const modelId = typeof location === 'undefined' ? 'xor' : new URLSearchParams(location.search).get('model') || 'xor';
if (modelId !== 'xor') {
  $('provider').value = 'guided';
  $('provider').querySelector('option[value="demo"]').disabled = true;
  document.querySelector('.model-tag').textContent = 'MNIST MLP · 50,890 parameters';
}
const requestedProvider = typeof location === 'undefined' ? null : new URLSearchParams(location.search).get('provider');
if (['guided', 'random', 'openai', ...(modelId === 'xor' ? ['demo'] : [])].includes(requestedProvider)) $('provider').value = requestedProvider;
const sessionKey = `weightforge-session-${modelId}`;
const configIds = ['provider', 'budget', 'seed', 'random-runs', 'max-delta', 'max-changed'];

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}

async function request(path, options = {}) {
  const response = await fetch('/api' + path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`);
  }
  return response.json();
}

function setConnection(ok, label) {
  connected = ok;
  $('connection').textContent = label;
  $('connection').classList.toggle('online', ok);
  $('start').disabled = !ok || isBusy(state);
}

function subscribe(id) {
  if (source) source.close();
  source = new EventSource(`/api/sessions/${encodeURIComponent(id)}/events?after=${state.lastId}`);
  source.onopen = () => setConnection(true, 'Connected · local CPU');
  source.addEventListener('update', message => {
    const event = JSON.parse(message.data);
    state = applyEvent(state, event);
    if (['accept', 'rollback'].includes(event.stage)) selected = event;
    render();
    if (['completed', 'cancelled', 'failed'].includes(event.stage)) {
      source.close();
      if (event.stage === 'failed') showError(`Experiment failed: ${event.message}`);
    }
  });
  source.onerror = async () => {
    if (!isBusy(state) && state.status !== 'idle') return;
    setConnection(false, 'Reconnecting…');
    try {
      const run = await request(`/sessions/${encodeURIComponent(id)}`);
      if (run.status === 'completed' && state.lastId >= run.event_count) source.close();
    } catch (error) {
      source.close();
      state.status = 'failed';
      render();
      showError(`${error.message}. Start a fresh experiment after reconnecting.`);
    }
  };
}

$('controls').addEventListener('submit', async event => {
  event.preventDefault();
  if (isBusy(state)) return;
  showError('');
  $('start').disabled = true;
  const config = { provider: $('provider').value, llm_consent: Boolean(document.getElementById('llm-consent')?.checked), budget: Number($('budget').value),
    model_id: modelId,
    seed: Number($('seed').value), random_runs: Number($('random-runs').value),
    max_delta: Number($('max-delta').value), max_changed: Number($('max-changed').value) };
  try {
    const run = await request('/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config) });
    state = initialState();
    state.status = 'queued';
    sessionId = run.id;
    selected = null;
    reviewIndex = 0;
    try { localStorage.setItem(sessionKey, sessionId); } catch { /* Private browsing. */ }
    render();
    subscribe(sessionId);
  } catch (error) {
    showError(error.message);
    $('start').disabled = !connected;
  }
});

$('stop').addEventListener('click', async () => {
  $('stop').disabled = true;
  try {
    await request(`/sessions/${encodeURIComponent(sessionId)}/stop`, { method: 'POST' });
    $('event-message').textContent = 'Stop requested; finishing the current safe operation.';
  } catch (error) { showError(error.message); $('stop').disabled = false; }
});

$('provider').addEventListener('change', () => {
  $('provider-note').textContent = $('provider').value === 'demo'
    ? 'Scripted XOR walkthrough: lower the bias, then test its reversal. This is a demonstration, not discovery.'
    : ({guided:'Guided calibration corrects validation output-probability bias. No LLM or backpropagation.',random:'Seeded random sparse edits; improvements are not guaranteed.',openai:'Sends bounded validation observations to OpenAI. API charges apply; no holdout is sent.'}[$('provider').value]);
});

for (const [id, value] of [['show-loss', 'loss'], ['show-accuracy', 'accuracy']]) {
  $(id).addEventListener('click', () => {
    metric = value;
    for (const key of ['loss', 'accuracy']) {
      $(`show-${key}`).classList.toggle('selected', key === value);
      $(`show-${key}`).setAttribute('aria-pressed', String(key === value));
    }
    drawChart();
  });
}

$('review').addEventListener('click', () => {
  if (!state.decisions.length) return;
  selected = state.decisions[reviewIndex % state.decisions.length];
  reviewIndex++;
  renderInspector();
  renderHistory();
  $('review').textContent = `Review edit ${(reviewIndex % state.decisions.length) + 1}`;
});

function render() {
  const busy = isBusy(state);
  configIds.forEach(id => { $(id).disabled = busy; });
  $('start').disabled = busy || !connected;
  $('start').lastChild.textContent = busy ? ' Experiment running' : ' Start experiment';
  $('stop').disabled = !busy;
  $('review').disabled = busy || !state.decisions.length;
  $('progress').max = Number($('budget').value);
  $('progress').value = state.step;
  $('progress-label').textContent = state.initial ? `Proposal ${state.step} / ${$('budget').value}` : 'No run started';
  $('elapsed').textContent = `${(state.elapsed / 1000).toFixed(2)} s elapsed`;
  const labels = { idle: 'Ready to experiment', queued: 'Queued for the worker', running: 'Experiment in progress',
    completed: 'Experiment complete', cancelled: 'Experiment stopped', failed: 'Experiment failed' };
  $('status').textContent = state.stage === 'baseline' ? 'Comparing random trials' : labels[state.status];
  document.querySelectorAll('#stages li').forEach(li => {
    const active = ['accept', 'rollback'].includes(state.stage) ? 'decision' :
      state.stage === 'baseline' ? 'completed' : state.stage;
    li.classList.toggle('current', li.dataset.stage === active);
  });
  if (state.retained) {
    $('loss').textContent = number(state.retained.loss);
    const improvement = (state.initial.loss - state.retained.loss) / state.initial.loss * 100;
    $('loss-change').textContent = `${improvement.toFixed(1)}% lower`;
    $('loss-before').textContent = `Started at ${number(state.initial.loss)} · lower is better`;
    $('accuracy').textContent = percent(state.retained.accuracy);
    $('accuracy-change').textContent = `+${((state.retained.accuracy - state.initial.accuracy) * 100).toFixed(1)} pts`;
    $('accuracy-before').textContent = `Started at ${percent(state.initial.accuracy)} · ${modelId === 'xor' ? 128 : 1000} validation points`;
  } else {
    $('loss').textContent = $('accuracy').textContent = '—';
    $('loss-change').textContent = 'Waiting for first run';
    $('accuracy-change').textContent = 'No regression allowed';
    $('loss-before').textContent = 'An edit must reduce loss to stay.';
    $('accuracy-before').textContent = `Correct predictions on ${modelId === 'xor' ? 128 : 1000} validation points.`;
  }
  const accepted = state.decisions.filter(row => row.accepted).length;
  $('accepted').replaceChildren(document.createTextNode(`${accepted} `), Object.assign(document.createElement('small'), { textContent: `/ ${state.decisions.length}` }));
  $('rollback-count').textContent = `${state.decisions.length - accepted} rolled back`;
  const last = state.events.at(-1);
  $('event-message').textContent = last ? `${last.stage.toUpperCase()} · ${last.message || 'Event recorded'}` : 'Waiting for a real experiment event.';
  $('event-count').textContent = `${state.events.length} events`;
  $('download').classList.toggle('disabled', state.status !== 'completed');
  $('download').setAttribute('aria-disabled', String(state.status !== 'completed'));
  if (state.status === 'completed') $('download').href = `/api/sessions/${sessionId}/results`;
  else $('download').removeAttribute('href');
  drawChart();
  renderInspector();
  renderHistory();
  renderComparison();
}

function renderInspector() {
  const row = selected;
  const change = row?.changes?.[0] || state.changes[0];
  const edit = row?.proposal?.edits?.[0] || state.proposal?.edits?.[0];
  $('coordinate').textContent = edit ? `${edit.parameter} [${edit.index}]` : '—';
  $('weight-before').textContent = change ? number(change.before) : '—';
  $('weight-after').textContent = change ? number(change.candidate) : '—';
  $('edit-reason').textContent = row ? `Edit ${row.step}: ${row.reason}` : state.proposal?.reason || 'Every candidate must pass the same validation rule.';
  const tag = $('decision-tag');
  tag.textContent = row ? row.accepted ? 'Accepted' : row.error ? 'Invalid' : 'Rolled back' : 'Idle';
  tag.className = `pill ${row ? row.accepted ? 'accept' : 'rollback' : ''}`;
  $('undo-bytes').textContent = state.events.length ? `${state.undoBytes} B` : '—';
  $('origin-bytes').textContent = state.events.length ? `${state.originBytes} B` : '—';
}

function renderHistory() {
  const body = $('history-rows');
  body.replaceChildren();
  if (!state.decisions.length) {
    const cell = document.createElement('td');
    cell.colSpan = 6; cell.className = 'empty';
    cell.textContent = isBusy(state) ? 'Waiting for the first measured decision…' : 'Your next experiment starts here. No edits have been made.';
    const row = document.createElement('tr');
    row.append(cell);
    body.append(row);
    return;
  }
  for (const row of [...state.decisions].reverse()) {
    const tr = document.createElement('tr');
    tr.className = 'history-row'; tr.tabIndex = 0;
    tr.setAttribute('aria-selected', String(selected?.step === row.step));
    const choose = () => { selected = row; renderInspector(); renderHistory(); };
    tr.addEventListener('click', choose);
    tr.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
    const edit = row.proposal.edits[0];
    const values = [String(row.step).padStart(2, '0'), `${edit.parameter}[${edit.index}] ${edit.delta >= 0 ? '+' : ''}${number(edit.delta)}`,
      number(row.candidate?.loss), number(row.retained.loss), row.accepted ? 'Accepted' : row.error ? 'Invalid' : 'Rolled back', row.reason];
    values.forEach((value, i) => {
      const cell = document.createElement('td');
      if (i === 4) {
        const pill = document.createElement('span'); pill.className = `pill ${row.accepted ? 'accept' : 'rollback'}`;
        pill.textContent = value; cell.append(pill);
      } else { cell.textContent = value; if (i === 1) cell.className = 'edit-code'; }
      tr.append(cell);
    });
    body.append(tr);
  }
}

const svgNS = 'http://www.w3.org/2000/svg';
function svgElement(name, attrs, text) {
  const element = document.createElementNS(svgNS, name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function drawChart() {
  const svg = $('chart');
  svg.replaceChildren();
  const data = chartData(state, metric);
  const left = 54, right = 692, top = 22, bottom = 238;
  const maxStep = Math.max(Number($('budget').value) || 12, state.step);
  const values = [...data.retained, ...data.candidates].map(p => p.y);
  let min = metric === 'accuracy' ? 0 : Math.max(0, Math.min(...values, 0) * .9);
  let max = metric === 'accuracy' ? 1 : Math.max(...values, .5) * 1.12;
  if (min === max) max = min + 1;
  const x = value => left + value / maxStep * (right - left);
  const y = value => bottom - (value - min) / (max - min) * (bottom - top);
  for (let i = 0; i < 5; i++) {
    const value = min + (max - min) * i / 4;
    svg.append(svgElement('line', { x1: left, x2: right, y1: y(value), y2: y(value), stroke: '#eaf0e4', 'stroke-dasharray': '3 5' }));
    svg.append(svgElement('text', { x: left - 12, y: y(value) + 4, 'text-anchor': 'end', fill: '#98a48c', 'font-size': 10 }, metric === 'accuracy' ? `${Math.round(value * 100)}%` : value.toFixed(2)));
  }
  const tickCount = Math.min(6, maxStep);
  for (let i = 0; i <= tickCount; i++) {
    const tick = Math.round(i / tickCount * maxStep);
    svg.append(svgElement('text', { x: x(tick), y: bottom + 24, 'text-anchor': 'middle', fill: '#98a48c', 'font-size': 10 }, tick));
  }
  if (!data.retained.length) {
    svg.append(svgElement('text', { x: 375, y: 128, 'text-anchor': 'middle', fill: '#899c7a', 'font-size': 13 }, 'Start a run to see measured results.'));
    return;
  }
  svg.append(svgElement('line', { x1: left, x2: right, y1: y(data.baseline), y2: y(data.baseline), stroke: '#b3bda9', 'stroke-dasharray': '5 5', 'stroke-width': 1.5 }));
  const path = data.retained.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p.y)}`).join(' ');
  svg.append(svgElement('path', { d: `${path} L${x(data.retained.at(-1).x)},${bottom} L${left},${bottom} Z`, fill: '#348459', opacity: '.045' }));
  svg.append(svgElement('path', { d: path, fill: 'none', stroke: '#31805b', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }));
  for (const point of data.candidates) {
    const dot = svgElement('circle', { cx: x(point.x), cy: y(point.y), r: 4.5, fill: '#fffaf0', stroke: '#c89650', 'stroke-width': 1.6 });
    dot.append(svgElement('title', {}, `Proposal ${point.x} candidate ${metric}: ${point.y.toFixed(6)} · ${point.accepted ? 'accepted' : 'rolled back'}`));
    dot.addEventListener('click', () => { selected = state.decisions.find(r => r.step === point.x); renderInspector(); renderHistory(); });
    svg.append(dot);
  }
  for (const point of data.retained) {
    const dot = svgElement('circle', { cx: x(point.x), cy: y(point.y), r: 3.2, fill: '#31805b', stroke: 'white', 'stroke-width': 1 });
    dot.append(svgElement('title', {}, `Step ${point.x} retained ${metric}: ${point.y.toFixed(6)}`));
    svg.append(dot);
  }
  svg.setAttribute('aria-label', `${metric === 'loss' ? 'Validation loss' : 'Validation accuracy'}: retained ${data.retained.at(-1).y.toFixed(4)}, starting ${data.baseline.toFixed(4)}. Detailed values are in the edit history.`);
  $('chart-note').textContent = metric === 'loss' ? 'Lower loss is better. Hover over points for measured values.' : 'Higher accuracy is better. Rejected edits leave the retained curve unchanged.';
}

function renderComparison() {
  const target = $('comparison-content');
  if (!state.result) {
    $('baseline-distribution')?.remove();
    $('holdout-tag').textContent = state.status === 'cancelled' ? 'Not evaluated' : 'Holdout sealed';
    if (!target.classList.contains('comparison-empty')) {
      target.className = 'comparison-empty'; target.replaceChildren();
      const text = document.createElement('p'); text.textContent = 'Final holdout performance and the complete random distribution appear after all searches finish.'; target.append(text);
    }
    return;
  }
  $('holdout-tag').textContent = 'Final evaluation';
  target.className = 'comparison-results'; target.replaceChildren();
  const result = state.result, summary = baselineSummary(result);
  for (const [label, loss, detail] of [
    ['NO EDIT · HOLDOUT LOSS', result.unedited_holdout.loss, `${percent(result.unedited_holdout.accuracy)} accuracy`],
    ['RETAINED MODEL · HOLDOUT LOSS', result.holdout.loss, `${percent(result.holdout.accuracy)} accuracy`],
    ['RANDOM BASELINES · MEAN LOSS', summary.mean, `${summary.losses.length} trials · range ${number(summary.min)}–${number(summary.max)}`],
  ]) {
    const card = document.createElement('div'); card.className = 'comparison-result';
    for (const [tag, text] of [['span', label], ['strong', number(loss)], ['small', detail]]) card.append(Object.assign(document.createElement(tag), { textContent: text }));
    target.append(card);
  }
  let list = $('baseline-distribution');
  if (!list) { list = document.createElement('div'); list.id = 'baseline-distribution'; list.className = 'baseline-list'; target.after(list); }
  list.textContent = `Every random trial (holdout loss): ${result.random_baselines.map((r, i) => `#${i + 1} seed ${r.seed}: ${number(r.holdout.loss)}`).join(' · ')}. Same one-edit proposals, budget and limits. Calibration uses output residuals; random search samples visible prefix coordinates. These are different search strategies, not evidence of general model repair.`;
}

async function initialize() {
  $('provider').dispatchEvent(new Event('change'));
  if (modelId !== 'xor') document.querySelector('.comparison .muted').textContent = 'The separate 2,000-example MNIST test subset is evaluated after all searches finish. All trials start from your imported checkpoint.';
  try {
    await request('/health');
    setConnection(true, 'Connected · local CPU');
    let saved;
    try { saved = localStorage.getItem(sessionKey); } catch { /* Optional persistence. */ }
    if (saved && !requestedProvider) {
      try {
        const run = await request(`/sessions/${encodeURIComponent(saved)}`);
        sessionId = run.id;
        const map = { 'provider': 'provider', 'budget': 'budget', 'seed': 'seed', 'random-runs': 'random_runs', 'max-delta': 'max_delta', 'max-changed': 'max_changed' };
        Object.entries(map).forEach(([id, key]) => { $(id).value = run.config[key]; });
        $('provider').dispatchEvent(new Event('change'));
        state.status = 'queued';
        subscribe(saved);
      } catch { try { localStorage.removeItem(sessionKey); } catch { /* Optional. */ } }
    }
  } catch {
    setConnection(false, 'Service unavailable');
    showError('The experiment service is unavailable. Start the FastAPI app or Docker Compose, then reload this page.');
  }
  render();
}

initialize();
