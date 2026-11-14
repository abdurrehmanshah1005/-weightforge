// Lightweight DOM harness for wiring/rendering. Server events are explicit test
// fixtures, not inference results. Browser visual QA is a separate check.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import * as labState from '../frontend/public/lab-state.mjs';

const code = fs.readFileSync(new URL('../frontend/public/app.js', import.meta.url), 'utf8')
  .replace(/^import .*?;\r?\n/, 'const { initialState, applyEvent, chartData, baselineSummary, isBusy } = labState;\n');
const html = fs.readFileSync(new URL('../frontend/public/experiment.html', import.meta.url), 'utf8');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.attributes = {}; this.listeners = {};
    this.textContent = ''; this.disabled = false; this.value = ''; this.dataset = {};
    const classes = new Set();
    this.classList = { toggle(key, on) { on ? classes.add(key) : classes.delete(key); }, contains(key) { return classes.has(key); } };
    this.lastChild = { textContent: '' };
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  remove() { this.removed = true; }
  after() {}
  dispatchEvent(event) { this.listeners[event.type]?.(event); }
}

async function harness(healthy = true) {
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  for (const [key, value] of Object.entries({ provider: 'guided', budget: '2', seed: '42', 'random-runs': '1', 'max-delta': '2', 'max-changed': '8' })) elements.get(key).value = value;
  const streams = [];
  class Source {
    constructor() { this.handlers = {}; streams.push(this); }
    close() { this.closed = true; }
    addEventListener(type, handler) { this.handlers[type] = handler; }
    emit(event) { this.handlers.update({ data: JSON.stringify(event) }); }
  }
  const document = { getElementById: key => elements.get(key), querySelectorAll: () => [],
    createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag),
    createTextNode: text => ({ textContent: text }) };
  const context = { document, labState, EventSource: Source, localStorage: { getItem: () => null, setItem() {} },
    Event: class { constructor(type) { this.type = type; } },
    fetch: async path => ({ ok: healthy || path !== '/api/health', status: healthy ? 200 : 503,
      json: async () => path === '/api/sessions' ? { id: 'test-session' } : {} }) };
  vm.runInNewContext(code, context, { filename: 'app.js' });
  await new Promise(resolve => setImmediate(resolve));
  return { elements, streams };
}

test('dashboard initializes without crashing and blocks unavailable inference', async () => {
  const { elements } = await harness(false);
  assert.equal(elements.get('connection').textContent, 'Service unavailable');
  assert.equal(elements.get('start').disabled, true);
  assert.equal(elements.get('history-rows').children[0].children[0].colSpan, 6);
  assert.match(elements.get('error').textContent, /unavailable/);
});

test('controls, SSE, decision table, review, export and reset are wired together', async () => {
  const { elements: e, streams } = await harness();
  await e.get('controls').listeners.submit({ preventDefault() {} });
  assert.equal(streams.length, 1);
  const stream = streams[0];
  stream.onopen();
  stream.emit({ id: 1, stage: 'started', elapsed_ms: 1, initial: { loss: .4, accuracy: .6 }, model: {} });
  const proposal = { edits: [{ parameter: '2.bias', index: 0, delta: -.5 }] };
  const row = { step: 1, accepted: true, proposal, reason: 'Validation improved', candidate: { loss: .2, accuracy: 1 },
    retained: { loss: .2, accuracy: 1 }, changes: [{ before: 0, candidate: -.5 }], elapsed_ms: 2 };
  stream.emit({ id: 2, stage: 'validate', undo_bytes: 4 });
  stream.emit({ id: 3, stage: 'accept', ...row, retained_origin_bytes: 4 });
  stream.emit({ id: 4, stage: 'rollback', ...row, step: 2, accepted: false, candidate: { loss: .4, accuracy: .6 }, reason: 'Restored weights' });
  assert.equal(e.get('loss').textContent, '0.2000');
  assert.equal(e.get('history-rows').children.length, 2);
  assert.equal(e.get('decision-tag').textContent, 'Rolled back');
  const result = { holdout: { loss: .19, accuracy: 1 }, unedited_holdout: { loss: .39, accuracy: .6 },
    random_baselines: [{ seed: 142, holdout: { loss: .3 } }] };
  stream.emit({ id: 5, stage: 'completed', elapsed_ms: 5, result });
  assert.equal(stream.closed, true);
  assert.equal(e.get('start').disabled, false);
  assert.equal(e.get('download').href, '/api/sessions/test-session/results');
  assert.equal(e.get('comparison-content').children.length, 3);
  e.get('review').listeners.click();
  assert.equal(e.get('decision-tag').textContent, 'Accepted');
  await e.get('controls').listeners.submit({ preventDefault() {} });
  assert.equal(e.get('loss').textContent, '—');
  assert.equal(e.get('comparison-content').children.length, 1);
});

test('switching chart metrics changes its accessible label', async () => {
  const { elements: e, streams } = await harness();
  await e.get('controls').listeners.submit({ preventDefault() {} });
  streams[0].emit({ id: 1, stage: 'started', elapsed_ms: 1, initial: { loss: .4, accuracy: .6 } });
  e.get('show-accuracy').listeners.click();
  assert.match(e.get('chart').attributes['aria-label'], /Validation accuracy/);
  assert.equal(e.get('show-accuracy').attributes['aria-pressed'], 'true');
});
