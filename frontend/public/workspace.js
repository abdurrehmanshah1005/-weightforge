const page = document.body.dataset.page;
const $ = id => document.getElementById(id);
function el(tag, text, className) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (className) n.className = className; return n; }
function link(text, href, style='quiet') { const n = el('a', text, style); n.href = href; return n; }
async function api(path, options) {
  const response = await fetch('/api' + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed');
  return data;
}
function error(message) { const box=$('service-message'); box.hidden=false; box.textContent=message; box.className='notice error'; }
function empty(target, title, description, action) {
  const box=el('div',undefined,'empty-state'); box.append(el('div','◈','empty-icon'),el('strong',title),el('p',description)); if(action)box.append(action); target.replaceChildren(box);
}
function modelRows(rows, target) {
  target.replaceChildren();
  for (const row of rows) {
    const card=el('div',undefined,'model-card'); const detail=el('div',undefined,'details');
    detail.append(el('h3',row.name || 'MNIST classifier'),el('p',`MNIST MLP · 50,890 parameters · ${(row.bytes/1024).toFixed(0)} KiB · ${row.id.slice(0,8)}`));
    card.append(el('div','⠿','model-symbol'), detail, link('Create experiment →',`/experiment?model=${row.id}`)); target.append(card);
  }
  if(!rows.length)empty(target,'Your first model belongs here.','Train the supported MNIST classifier, then import its checkpoint to start experimenting.');
}
function runRows(rows, target) {
  target.replaceChildren();
  for(const row of rows) {
    const card=el('div',undefined,'model-card'), detail=el('div',undefined,'details');
    detail.append(el('h3',`${row.config.model_id === 'xor' ? 'XOR' : 'MNIST'} · ${row.config.provider}`),el('p',`Validation loss ${row.initial.loss.toFixed(4)} → ${row.retained.loss.toFixed(4)} · seed ${row.config.seed} · ${row.id.slice(0,8)}`));
    card.append(el('div','⌁','model-symbol'),detail,link('View report →',`/report?id=${row.id}`)); target.append(card);
  }
  if(!rows.length)empty(target,'Let the evidence accumulate.','Completed experiments appear here with their retained metrics and downloadable checkpoints.',link('Start an experiment','/experiment'));
}
async function report() {
  const id=new URLSearchParams(location.search).get('id');
  if(!/^[a-f0-9]{32}$/.test(id || ''))throw new Error('Choose an experiment from Saved runs.');
  const result=await api(`/experiments/${id}`);
  $('report-title').textContent=`${result.task?.name || result.config.model_id} · ${result.config.provider}`;
  $('report-disclosure').textContent=result.provider_disclosure;
  const rows=[['Validation loss',result.initial.loss,result.retained.loss],['Validation accuracy',result.initial.accuracy,result.retained.accuracy],['Holdout loss',result.unedited_holdout.loss,result.holdout.loss],['Holdout accuracy',result.unedited_holdout.accuracy,result.holdout.accuracy]];
  for(const row of rows){const tr=el('tr'); row.forEach(x=>tr.append(el('td',typeof x==='number'?x.toFixed(4):x))); $('report-metrics').append(tr);}
  $('report-actions').append(link('Download checkpoint',`/api/experiments/${id}/checkpoint`,'primary'),link('Download full JSON',`/api/experiments/${id}/download`));
  $('report-patch').textContent=JSON.stringify(result.patch,null,2);
  $('report-baselines').textContent=result.random_baselines.map((r,i)=>`Trial ${i+1} · seed ${r.seed} · holdout loss ${r.holdout.loss.toFixed(4)}`).join('\n');
}
async function initialize() {
  document.querySelectorAll('[data-nav]').forEach(a=>a.classList.toggle('active',a.dataset.nav===page));
  try {
    const capability=await api('/capabilities');
    $('connection').textContent='Backend connected'; $('connection').className='connection online';
    if($('data-status'))$('data-status').textContent=capability.mnist_data_ready?'Dataset ready':'Dataset preparation needed';
    if($('llm-status'))$('llm-status').textContent=capability.llm_configured?`Configured: ${capability.llm_model}`:'Not configured';
    if(page==='overview') {
      const [models,runs]=await Promise.all([api('/models'),api('/experiments')]);
      $('model-count').textContent=models.length; $('run-count').textContent=runs.length;
      $('provider-count').textContent=capability.llm_configured?'4':'3'; runRows(runs.slice(0,3),$('recent-runs'));
    }
    if(page==='models')modelRows(await api('/models'),$('model-list'));
    if(page==='runs')runRows(await api('/experiments'),$('run-list'));
    if(page==='report')await report();
  } catch(e) { $('connection').textContent='Backend unavailable'; error(e.message || 'Start the backend to load your workspace.'); }
}
if($('upload-form'))$('upload-form').addEventListener('submit',async event=>{
  event.preventDefault();const file=$('checkpoint').files[0];if(!file || file.size>2000000){error('Select a checkpoint no larger than 2 MB.');return;}
  $('upload-button').disabled=true;$('upload-status').textContent='Validating tensor names, shapes and values…';
  try {await api('/models',{method:'POST',body:file,headers:{'X-Model-Name':encodeURIComponent($('model-name').value || file.name)}});$('upload-status').textContent='Imported successfully. Select your model below.';modelRows(await api('/models'),$('model-list'));}
  catch(e){error(e.message);$('upload-status').textContent='Import was not completed.';}finally{$('upload-button').disabled=false;}
});
initialize();
