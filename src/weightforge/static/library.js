const results = location.pathname === '/results';
const items = document.getElementById('items');
async function refresh() {
  const response = await fetch(results ? '/experiments' : '/models');
  if (!response.ok) throw new Error('Could not load the library');
  const rows = await response.json();
  items.replaceChildren();
  if (!rows.length) items.textContent = results ? 'No completed experiments yet.' : 'No imported checkpoints yet. XOR remains available in the experiment lab.';
  for (const row of rows) {
    const card = document.createElement('p');
    if (results) card.textContent = `${row.id} · ${row.config.model_id || 'xor'} · loss ${row.initial.loss.toFixed(4)} → ${row.retained.loss.toFixed(4)}`;
    else {
      const link = document.createElement('a');
      link.href = `/?model=${row.id}`;
      link.className = 'quiet';
      link.textContent = `MNIST MLP · ${row.id.slice(0, 8)} · ${(row.bytes / 1024).toFixed(0)} KiB · Run experiment →`;
      card.append(link);
    }
    items.append(card);
  }
}
if (results) {
  document.getElementById('title').textContent = 'Saved results';
  document.getElementById('list-title').textContent = 'Completed experiments';
  document.getElementById('import-panel').hidden = true;
}
document.getElementById('upload').addEventListener('submit', async event => {
  event.preventDefault();
  const message = document.getElementById('message');
  const file = document.getElementById('checkpoint').files[0];
  if (!file || file.size > 2000000) { message.textContent = 'Choose a checkpoint under 2 MB.'; return; }
  message.textContent = 'Validating checkpoint…';
  try {
    const response = await fetch('/models', { method: 'POST', body: file });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail);
    message.textContent = 'Checkpoint imported. Select it below to begin.';
    await refresh();
  } catch (error) { message.textContent = error.message; }
});
refresh().catch(error => { items.textContent = error.message; });
