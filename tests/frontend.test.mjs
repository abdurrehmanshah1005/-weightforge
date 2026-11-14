import test from 'node:test';
import assert from 'node:assert/strict';
import { initialState, applyEvent, chartData, baselineSummary, isBusy } from '../frontend/public/lab-state.mjs';

test('candidate evaluation never replaces the retained model', () => {
  let state = applyEvent(initialState(), { id: 1, stage: 'started', elapsed_ms: 1, initial: { loss: .4, accuracy: .6 } });
  const before = state;
  state = applyEvent(state, { id: 2, stage: 'evaluate', elapsed_ms: 2, candidate: { loss: .8, accuracy: .3 } });
  assert.equal(state.retained.loss, .4);
  assert.equal(before.events.length, 1);
  assert.equal(isBusy(state), true);
});

test('rejected edits stay visible without corrupting retained chart', () => {
  let state = applyEvent(initialState(), { id: 1, stage: 'started', initial: { loss: .4, accuracy: .6 } });
  state = applyEvent(state, { id: 2, stage: 'accept', step: 1, accepted: true, candidate: { loss: .2, accuracy: 1 }, retained: { loss: .2, accuracy: 1 } });
  state = applyEvent(state, { id: 3, stage: 'rollback', step: 2, accepted: false, candidate: { loss: .4, accuracy: .6 }, retained: { loss: .2, accuracy: 1 } });
  assert.deepEqual(chartData(state).retained.map(p => p.y), [.4, .2, .2]);
  assert.deepEqual(chartData(state).candidates.map(p => p.y), [.2, .4]);
  assert.deepEqual(chartData(state, 'accuracy').retained.map(p => p.y), [.6, 1, 1]);
  assert.equal(state.decisions.length, 2);
});

test('reconnect duplicates are ignored', () => {
  const event = { id: 1, stage: 'queued' };
  const state = applyEvent(initialState(), event);
  assert.equal(applyEvent(state, event), state);
});

test('holdout appears only in the completed result', () => {
  let state = applyEvent(initialState(), { id: 1, stage: 'baseline', trial: 1, validation: { loss: .3 } });
  assert.equal(state.result, null);
  const result = { holdout: { loss: .2 }, random_baselines: [{ holdout: { loss: .1 } }, { holdout: { loss: .5 } }] };
  state = applyEvent(state, { id: 2, stage: 'completed', result });
  assert.equal(state.result, result);
  assert.equal(isBusy(state), false);
  assert.equal(baselineSummary(result).mean, .3);
  assert.deepEqual(baselineSummary(result).losses, [.1, .5]);
});

test('invalid proposals can be charted without candidate metrics', () => {
  let state = applyEvent(initialState(), { id: 1, stage: 'started', initial: { loss: .4, accuracy: .6 } });
  state = applyEvent(state, { id: 2, stage: 'rollback', step: 1, error: 'invalid', retained: { loss: .4, accuracy: .6 } });
  assert.deepEqual(chartData(state).candidates, []);
  assert.equal(chartData(state).retained.length, 2);
});

test('terminal states release controls and never fabricate results', () => {
  for (const stage of ['failed', 'cancelled']) {
    const state = applyEvent(initialState(), { id: 1, stage });
    assert.equal(isBusy(state), false);
    assert.equal(state.result, null);
  }
});
