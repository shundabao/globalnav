const state = {
  examples: [],
  selected: null,
  offset: 0,
  limit: 25,
  total: 0,
};

const $ = (id) => document.getElementById(id);

function pretty(obj) {
  if (obj === null || obj === undefined) return 'null';
  return JSON.stringify(obj, null, 2);
}

async function loadExamples(reset = true) {
  if (reset) state.offset = 0;
  const params = new URLSearchParams({
    offset: state.offset,
    limit: state.limit,
  });
  const split = $('splitFilter').value;
  const q = $('searchBox').value.trim();
  if (split) params.set('split', split);
  if (q) params.set('q', q);
  const resp = await fetch(`/api/bench/examples?${params.toString()}`);
  const data = await resp.json();
  if (!resp.ok) {
    $('countLabel').textContent = data.error || 'Load failed';
    return;
  }
  state.examples = data.examples || [];
  state.total = data.total || 0;
  $('countLabel').textContent = `${state.offset + 1}-${Math.min(state.offset + state.examples.length, state.total)} / ${state.total}`;
  renderList();
}

function renderList() {
  const root = $('exampleList');
  root.innerHTML = '';
  state.examples.forEach((ex) => {
    const btn = document.createElement('button');
    btn.className = `example-row ${state.selected && state.selected.id === ex.id ? 'active' : ''}`;
    btn.innerHTML = `
      <span class="row-id">${ex.id}</span>
      <span class="row-split">${ex.split}</span>
      <span class="row-text">${escapeHtml(ex.instruction)}</span>
    `;
    btn.addEventListener('click', () => selectExample(ex));
    root.appendChild(btn);
  });
}

function selectExample(ex) {
  state.selected = ex;
  $('emptyState').classList.add('hidden');
  $('detail').classList.remove('hidden');
  $('detailSplit').textContent = ex.split;
  $('detailId').textContent = ex.id;
  $('detailLang').textContent = ex.language;
  $('instructionText').textContent = ex.instruction;
  $('intentJson').textContent = pretty(ex.gold_intent);
  $('clarificationJson').textContent = pretty(ex.clarification);
  $('followerJson').textContent = pretty(ex.follower_annotation);
  $('rewrittenInstruction').value = ex.instruction || '';
  $('proceduralNotes').value = followerNotes(ex);
  $('comments').value = '';
  $('endpointOk').checked = false;
  $('routeOk').checked = false;
  $('clarificationOk').checked = false;
  $('followerOracleOk').checked = ex.split !== 'hybrid_follower';
  renderSegments(ex.route_annotation || {});
  renderList();
}

function renderSegments(route) {
  const root = $('segments');
  root.innerHTML = '';
  const segments = route.segments || [];
  if (!segments.length) {
    root.innerHTML = '<p class="muted">No route segments. This may be a clarification-only example.</p>';
    return;
  }
  segments.forEach((seg) => {
    const card = document.createElement('section');
    card.className = 'segment-card';
    card.innerHTML = `
      <div class="segment-title">
        <strong>${escapeHtml(seg.segment_id)}</strong>
        <span>${escapeHtml(seg.from || '')} → ${escapeHtml(seg.to || '')}</span>
      </div>
      <div class="chips">${(seg.allowed_modes || seg.mode_chain || []).map((m) => `<span>${escapeHtml(m)}</span>`).join('')}</div>
      <p class="muted">Evidence: ${(seg.evidence || []).map(escapeHtml).join(' · ') || 'none'}</p>
      <p class="muted">Status: ${escapeHtml(seg.annotation_status || 'unmarked')}</p>
    `;
    root.appendChild(card);
  });
}

function followerNotes(ex) {
  const follower = ex.follower_annotation || {};
  const notes = follower.procedural_notes || [];
  return notes.map((item) => {
    const lines = (item.notes || []).map((n) => `- ${n}`).join('\n');
    return `${item.segment_id}\n${lines}`;
  }).join('\n\n');
}

async function saveReview(event) {
  event.preventDefault();
  if (!state.selected) return;
  const payload = {
    example_id: state.selected.id,
    annotator: $('annotator').value.trim(),
    review_status: $('reviewStatus').value,
    endpoint_ok: $('endpointOk').checked,
    route_ok: $('routeOk').checked,
    clarification_ok: $('clarificationOk').checked,
    follower_oracle_ok: $('followerOracleOk').checked,
    rewritten_instruction: $('rewrittenInstruction').value,
    procedural_notes: $('proceduralNotes').value,
    comments: $('comments').value,
  };
  $('saveStatus').textContent = 'Saving...';
  const resp = await fetch('/api/bench/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  $('saveStatus').textContent = resp.ok ? `Saved ${data.review.example_id}` : (data.error || 'Save failed');
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

$('reloadBtn').addEventListener('click', () => loadExamples(true));
$('splitFilter').addEventListener('change', () => loadExamples(true));
$('searchBox').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') loadExamples(true);
});
$('prevBtn').addEventListener('click', () => {
  state.offset = Math.max(0, state.offset - state.limit);
  loadExamples(false);
});
$('nextBtn').addEventListener('click', () => {
  if (state.offset + state.limit < state.total) {
    state.offset += state.limit;
    loadExamples(false);
  }
});
$('reviewForm').addEventListener('submit', saveReview);

loadExamples(true);
