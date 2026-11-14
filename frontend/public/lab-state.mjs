// Pure presentation state. Metrics always originate in server events.
export function initialState() {
  return { status: 'idle', events: [], decisions: [], baselines: [], initial: null,
    retained: null, result: null, model: null, lastId: 0, elapsed: 0, step: 0,
    stage: '', proposal: null, changes: [], undoBytes: 0, originBytes: 0 };
}

export function applyEvent(state, event) {
  if (!Number.isInteger(event.id) || event.id <= state.lastId) return state;
  const next = { ...state, lastId: event.id, elapsed: event.elapsed_ms,
    stage: event.stage, events: [...state.events, event] };
  if (event.stage === 'queued') next.status = 'queued';
  if (event.stage === 'started') {
    Object.assign(next, { status: 'running', initial: event.initial,
      retained: event.initial, model: event.model });
  }
  if (event.proposal) next.proposal = event.proposal;
  if (event.changes) next.changes = event.changes;
  if (event.undo_bytes !== undefined) next.undoBytes = event.undo_bytes;
  if (event.retained_origin_bytes !== undefined) next.originBytes = event.retained_origin_bytes;
  if (['accept', 'rollback'].includes(event.stage)) {
    next.decisions = [...state.decisions, event];
    next.retained = event.retained;
    next.step = event.step;
  }
  if (event.stage === 'baseline') next.baselines = [...state.baselines, event];
  if (['completed', 'cancelled', 'failed'].includes(event.stage)) next.status = event.stage;
  if (event.stage === 'completed') next.result = event.result;
  return next;
}

export function chartData(state, metric = 'loss') {
  if (!state.initial) return { retained: [], candidates: [], baseline: null };
  return {
    retained: [{ x: 0, y: state.initial[metric] }, ...state.decisions.map(row =>
      ({ x: row.step, y: row.retained[metric] }))],
    candidates: state.decisions.filter(row => row.candidate).map(row =>
      ({ x: row.step, y: row.candidate[metric], accepted: row.accepted })),
    baseline: state.initial[metric],
  };
}

export function baselineSummary(result) {
  const losses = result.random_baselines.map(row => row.holdout.loss);
  return { losses, mean: losses.reduce((a, b) => a + b, 0) / losses.length,
    min: Math.min(...losses), max: Math.max(...losses) };
}

export const isBusy = state => ['queued', 'running'].includes(state.status);
