/* GLOBALNAV Instruction Follower — map, segment options, step playback */

const MODE_COLORS = {
  walk: '#38bdf8', drive: '#fbbf24', bus: '#a3e635', train: '#f472b6',
  tram: '#c084fc', fly: '#a78bfa', maritime: '#2dd4bf',
};

const state = {
  sessionId: null,
  plan: null,
  selections: {},
  trajectory: [],
  stepIndex: 0,
  playing: false,
  playTimer: null,
  map: null,
  routeLayers: [],
  marker: null,
  pendingClarify: null,
};

const $ = id => document.getElementById(id);

const ACTION_LABELS = {
  forward: 'Continue',
  left: 'Turn left',
  right: 'Turn right',
  turn_around: 'Turn around',
  u_turn: 'U-turn',
  stop: 'Stop',
  board: 'Board',
  takeoff: 'Take off',
  cruise: 'Cruise',
  land: 'Land',
  taxi: 'Taxi',
  arrive: 'Arrive',
  depart: 'Depart',
  dock: 'Dock',
};

function modeColor(mode) {
  return MODE_COLORS[mode] || '#94a3b8';
}

function actionLabel(action) {
  return ACTION_LABELS[action] || action.replace(/_/g, ' ');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function updateHeaderTotals(plan, selections = {}) {
  const boxes = document.querySelectorAll('.header-metrics strong');
  if (boxes.length < 2 || !plan?.segments?.length) return;
  let totalMin = 0;
  let totalKm = 0;
  plan.segments.forEach(seg => {
    const oid = selections[seg.segment_id] || seg.default_option_id;
    const opt = (seg.options || []).find(o => o.option_id === oid) || (seg.options || [])[0];
    if (!opt) return;
    totalMin += Number(opt.duration_min || 0);
    totalKm += Number(opt.distance_km || 0);
  });
  boxes[0].textContent = formatMinutes(totalMin);
  boxes[1].textContent = totalKm ? `${totalKm.toFixed(1)} km` : '-- --';
}

function formatMinutes(minutes) {
  if (!minutes) return '--:--';
  if (minutes < 60) return `${Math.round(minutes)} min`;
  const hours = minutes / 60;
  if (hours < 24) return `${hours.toFixed(1)} h`;
  return `${Math.floor(hours / 24)}d ${Math.round(hours % 24)}h`;
}

function renderObservationPanel(obs) {
  const summary = $('obsSummary');
  if (!summary) return;
  if (!obs || obs.done) {
    summary.innerHTML = `
      <div class="obs-pill"><span>Status</span><span>${escapeHtml(obs?.message || 'Ready')}</span></div>`;
    renderActionList([]);
    return;
  }
  summary.innerHTML = `
    <div class="obs-pill"><span>Mode</span><span>${escapeHtml(obs.mode || '-')}</span></div>
    <div class="obs-pill"><span>Facing</span><span>${escapeHtml(obs.facing || '?')}${obs.heading_deg != null ? ` · ${escapeHtml(obs.heading_deg)}°` : ''}</span></div>
    <div class="obs-pill"><span>Leg</span><span>${escapeHtml(obs.leg_index || '-')} / ${escapeHtml(obs.leg_total || '-')}</span></div>
    <div class="obs-pill"><span>Route</span><span>${escapeHtml(obs.from || '-')} → ${escapeHtml(obs.to || '-')}</span></div>
    <div class="obs-pill"><span>Progress</span><span>${escapeHtml(obs.progress || '-')}</span></div>`;
  renderActionList(obs.available_actions || []);
}

function renderActionList(actions) {
  const box = $('actionList');
  if (!box) return;
  if (!actions.length) {
    box.innerHTML = '<button type="button" disabled>Prepare route</button>';
    return;
  }
  box.innerHTML = actions.map((action, i) => (
    `<button type="button" class="${i === 0 ? 'primary-action' : ''}" title="Simulator action">${escapeHtml(actionLabel(action))}</button>`
  )).join('');
}

function showStreetViewPlaceholder(message, detail = '') {
  const frame = document.querySelector('.sv-frame');
  const img = $('svImage');
  const placeholder = $('svPlaceholder');
  if (!frame || !img || !placeholder) return;
  frame.classList.remove('has-image');
  img.removeAttribute('src');
  const safeMessage = escapeHtml(message || '暂无街景');
  const safeDetail = detail ? `<p>${escapeHtml(detail)}</p>` : '';
  placeholder.innerHTML = `<span>Street View</span><p>${safeMessage}</p>${safeDetail}`;
}

function showStreetViewImage(url, alt) {
  const frame = document.querySelector('.sv-frame');
  const img = $('svImage');
  const placeholder = $('svPlaceholder');
  if (!frame || !img || !placeholder) return;
  frame.classList.remove('has-image');
  placeholder.innerHTML = '<span>Street View</span><p>正在加载街景图像…</p>';
  img.onload = () => frame.classList.add('has-image');
  img.onerror = () => showStreetViewPlaceholder('街景图像加载失败', '请检查 Street View Static API 或当前位置覆盖情况');
  img.alt = alt || 'Street view';
  img.src = `${url}&_=${Date.now()}`;
}

function initMap() {
  if (state.map) return;
  state.map = L.map('map', { zoomControl: true }).setView([-33.87, 151.21], 12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 19,
  }).addTo(state.map);
  state.marker = L.circleMarker([0, 0], {
    radius: 8, color: '#fff', weight: 2, fillColor: '#3b82f6', fillOpacity: 0.9,
  });
}

function clearRouteLayers() {
  state.routeLayers.forEach(l => state.map.removeLayer(l));
  state.routeLayers = [];
}

function drawRouteGeometry(geo) {
  if (!state.map || !geo) return;
  clearRouteLayers();
  const bounds = [];
  (geo.segments || []).forEach(seg => {
    (seg.polylines || []).forEach(pl => {
      const latlngs = (pl.geometry || []).map(p => [p[0], p[1]]);
      if (!latlngs.length) return;
      const layer = L.polyline(latlngs, {
        color: modeColor(pl.mode), weight: pl.mode === 'fly' ? 3 : 4,
        opacity: 0.85, dashArray: pl.mode === 'fly' ? '6 8' : null,
      }).addTo(state.map);
      layer.bindTooltip(`${pl.mode}: ${pl.from} → ${pl.to}`);
      state.routeLayers.push(layer);
      bounds.push(...latlngs);
    });
  });
  if (bounds.length) state.map.fitBounds(bounds, { padding: [40, 40] });
}

function updateMarker(obs) {
  if (!obs || !obs.lat || !obs.lon) return;
  state.marker.setLatLng([obs.lat, obs.lon]).addTo(state.map);
  state.map.panTo([obs.lat, obs.lon], { animate: true, duration: 0.4 });
}

function updateStreetView(obs) {
  const panel = $('svPanel');
  const meta = $('svMeta');
  const sv = obs?.streetview || {};
  panel.classList.remove('hidden');
  renderObservationPanel(obs);

  if (sv.available && sv.image_url) {
    showStreetViewImage(sv.image_url, `Street view ${obs.facing || ''}`);
    meta.textContent = `${sv.source || 'imagery'} · ${obs.facing || '?'} · ${obs.osm_instruction || ''}`;
    return;
  }

  if (sv.reason === 'disabled') {
    showStreetViewPlaceholder('街景已关闭', '勾选「街景图像」后重新准备路线');
    meta.textContent = '街景已关闭（取消勾选「街景图像」）';
  } else if (sv.reason === 'api_not_enabled' || sv.reason === 'no_key') {
    showStreetViewPlaceholder('街景 API 未可用', sv.message || '请配置 GOOGLE_MAPS_API_KEY 或 MAPILLARY_ACCESS_TOKEN');
    meta.textContent = sv.message || '未配置街景 API';
  } else if (sv.reason?.endsWith('_phase')) {
    showStreetViewPlaceholder(`${obs.mode || '当前'} 阶段无街景`, `${obs.from || ''} → ${obs.to || ''}`);
    meta.textContent = `✈ ${obs.from} → ${obs.to} · ${sv.message || sv.phase || obs.mode}`;
  } else if (sv.reason === 'no_coverage') {
    showStreetViewPlaceholder('此位置无街景覆盖', `${obs.from || ''} → ${obs.to || ''}`);
    meta.textContent = sv.message || '此位置无街景覆盖';
  } else {
    showStreetViewPlaceholder(sv.message || '暂无街景');
    meta.textContent = sv.message || '暂无街景';
  }
}

function streetviewStatusLine(status) {
  if (!status) return '街景: 未知';
  if (!status.enabled) return '街景: 已关闭';
  if (!status.working) return `街景: ${status.message || '不可用'}`;
  return `街景: ${status.source || '可用'}`;
}

function renderSegmentOptions(plan) {
  const box = $('segmentOptions');
  if (!plan?.segments?.length) {
    box.innerHTML = '';
    box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');
  box.innerHTML = plan.segments.map(seg => {
    const sel = state.selections[seg.segment_id] || seg.default_option_id;
    const chips = seg.options.map(opt => {
      const active = opt.option_id === sel ? 'active' : '';
      const rec = opt.is_recommended ? ' recommended' : '';
      const duration = opt.duration_display || '';
      const distance = opt.distance_km != null ? `${opt.distance_km} km` : '';
      return `<button type="button" class="opt-chip${active}${rec}" data-seg="${seg.segment_id}" data-opt="${opt.option_id}" title="${opt.tooltip || ''}">
        <span class="opt-chip-label">${opt.label}</span>
        <span class="opt-chip-meta">${duration}${distance ? ` · ${distance}` : ''}</span>
      </button>`;
    }).join('');
    return `<div class="seg-block"><div class="seg-title">${seg.title}</div><div class="opt-row">${chips}</div></div>`;
  }).join('');

  box.querySelectorAll('.opt-chip').forEach(btn => {
    btn.onclick = async () => {
      const segId = btn.dataset.seg;
      const optId = btn.dataset.opt;
      state.selections[segId] = optId;
      renderSegmentOptions(plan);
      if (!state.sessionId) return;
      $('status').textContent = '更新路线选择…';
      const res = await fetch('/api/follower/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId, selections: state.selections }),
      });
      const data = await res.json();
      if (data.error) {
        $('status').textContent = '错误: ' + data.error;
        return;
      }
      state.plan = data.plan || state.plan;
      updateHeaderTotals(state.plan, state.selections);
      drawRouteGeometry(data.route_geometry);
      state.trajectory = [];
      state.stepIndex = 0;
      renderTrace();
      updateMarker(data.initial_observation);
      updateStreetView(data.initial_observation);
      $('status').textContent = `路线已更新 · ${data.execution_legs} 执行段`;
    };
  });
}

function renderTrace() {
  const trace = $('trace');
  if (!state.trajectory.length) {
    trace.innerHTML = '<div class="trace-empty">尚无步骤 — 点击「单步」或「播放」</div>';
    return;
  }
  trace.innerHTML = state.trajectory.map((t, i) => {
    const obs = t.observation || {};
    const cur = i === state.stepIndex ? ' trace-current' : '';
    const sv = obs.streetview?.available ? '📷' : '';
    return `<div class="trace-step${cur}" data-idx="${i}">
      <div class="trace-action">Step ${t.step}: ${t.action} ${sv}</div>
      <div>[${obs.mode}] ${obs.facing || '?'} — ${obs.osm_instruction || obs.to || ''}</div>
      <div class="trace-notes">${obs.procedural_notes || ''}</div>
    </div>`;
  }).join('');

  trace.querySelectorAll('.trace-step').forEach(el => {
    el.onclick = () => {
      state.stepIndex = parseInt(el.dataset.idx, 10);
      highlightStep();
    };
  });
  highlightStep();
}

function highlightStep() {
  document.querySelectorAll('.trace-step').forEach((el, i) => {
    el.classList.toggle('trace-current', i === state.stepIndex);
  });
  const t = state.trajectory[state.stepIndex];
  if (t) {
    updateMarker(t.observation);
    updateStreetView(t.observation);
    $('scrubber').value = state.stepIndex;
    $('stepLabel').textContent = `${state.stepIndex + 1} / ${state.trajectory.length}`;
  }
}

function preparePayload() {
  return {
    instruction: $('instruction').value.trim(),
    rule_based: $('ruleBased').checked,
    no_llm: $('noLlm').checked,
    streetview: $('streetview').checked,
  };
}

async function parseJsonResponse(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    if (res.status === 405) {
      throw new Error('服务器版本过旧，请重启: python scripts/run_gui.py --port 8765');
    }
    throw new Error(text.slice(0, 120) || `HTTP ${res.status}`);
  }
}

function showClarify(questions) {
  state.pendingClarify = questions[0];
  $('clarifyQuestion').textContent = state.pendingClarify.question;
  const optsEl = $('clarifyOptions');
  optsEl.innerHTML = '';
  $('clarifyInput').classList.add('hidden');
  if (state.pendingClarify.options?.length) {
    state.pendingClarify.options.forEach(opt => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = opt;
      btn.onclick = () => submitClarify(opt);
      optsEl.appendChild(btn);
    });
  } else {
    $('clarifyInput').classList.remove('hidden');
  }
  $('clarifyModal').classList.remove('hidden');
}

async function submitClarify(answer) {
  $('clarifyModal').classList.add('hidden');
  const payload = preparePayload();
  const res = await fetch('/api/follower/clarify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...payload,
      category: state.pendingClarify.category,
      question: state.pendingClarify.question,
      answer: answer || $('clarifyInput').value.trim(),
    }),
  });
  const data = await parseJsonResponse(res);
  handlePrepareResponse(data);
}

function handlePrepareResponse(data) {
  if (data.status === 'clarifying') {
    $('status').textContent = '需要确认一项信息';
    showClarify(data.questions);
    return;
  }
  if (data.error) {
    $('status').textContent = '错误: ' + data.error;
    return;
  }

  state.sessionId = data.session_id;
  state.plan = data.plan;
  state.selections = data.selections || {};
  updateHeaderTotals(state.plan, state.selections);
  initMap();
  drawRouteGeometry(data.route_geometry);
  renderSegmentOptions(data.plan);
  updateMarker(data.initial_observation);
  updateStreetView(data.initial_observation);

  $('meta').classList.remove('hidden');
  $('meta').innerHTML = `
    <div class="route-header">${data.origin} → ${data.destination}</div>
    <div class="meta-line">执行段: ${data.execution_legs} · ${streetviewStatusLine(data.streetview_status)}</div>
    <details><summary>分解结果</summary><pre>${JSON.stringify(data.decomposed, null, 2)}</pre></details>`;

  $('playback').classList.remove('hidden');
  $('scrubber').max = 0;
  $('scrubber').value = 0;
  renderTrace();
  $('status').textContent = '准备完成 — 可单步或播放';
}

async function prepare() {
  const instruction = $('instruction').value.trim();
  if (!instruction) return;

  stopPlay();
  state.trajectory = [];
  state.stepIndex = 0;
  $('prepareBtn').disabled = true;
  $('status').textContent = 'LLM 提取起终点 + 分解指令 + 构建环境路线…';
  $('playback').classList.add('hidden');
  $('meta').classList.add('hidden');
  showStreetViewPlaceholder('正在准备路线…', '街景会随当前位置更新');
  $('svMeta').textContent = '';
  renderObservationPanel(null);

  try {
    const res = await fetch('/api/follower/prepare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(preparePayload()),
    });
    handlePrepareResponse(await parseJsonResponse(res));
  } catch (e) {
    $('status').textContent = '失败: ' + e.message;
  } finally {
    $('prepareBtn').disabled = false;
  }
}

async function stepOnce() {
  if (!state.sessionId) return;
  $('stepBtn').disabled = true;
  try {
    const res = await fetch('/api/follower/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    const data = await res.json();
    if (data.error) {
      $('status').textContent = '错误: ' + data.error;
      return;
    }
    state.trajectory.push({
      step: data.step,
      action: data.action,
      observation: data.observation,
      done: data.done,
    });
    state.stepIndex = state.trajectory.length - 1;
    $('scrubber').max = Math.max(0, state.trajectory.length - 1);
    renderTrace();
    if (data.done) {
      $('status').textContent = `完成: success=${data.success}`;
      stopPlay();
    } else {
      $('status').textContent = `Step ${data.step}: ${data.action}`;
    }
  } finally {
    $('stepBtn').disabled = false;
  }
}

function stopPlay() {
  state.playing = false;
  if (state.playTimer) clearInterval(state.playTimer);
  state.playTimer = null;
  $('playBtn').textContent = '▶ 播放';
}

async function togglePlay() {
  if (state.playing) {
    stopPlay();
    return;
  }
  if (!state.sessionId) return;
  state.playing = true;
  $('playBtn').textContent = '⏸ 暂停';
  state.playTimer = setInterval(async () => {
    if (state.trajectory.length && state.trajectory[state.trajectory.length - 1].done) {
      stopPlay();
      return;
    }
    await stepOnce();
  }, 600);
}

async function refreshSvHint() {
  const el = $('svHint');
  if (!el) return;
  try {
    const res = await fetch(`/api/streetview/status?enabled=${$('streetview').checked ? 1 : 0}`);
    const s = await res.json();
    el.textContent = streetviewStatusLine(s);
  } catch {
    el.textContent = '';
  }
}

function bindUI() {
  $('prepareBtn').onclick = prepare;
  $('stepBtn').onclick = stepOnce;
  $('playBtn').onclick = togglePlay;
  $('clarifySubmit').onclick = () => submitClarify();
  $('streetview').onchange = refreshSvHint;
  $('scrubber').oninput = () => {
    state.stepIndex = parseInt($('scrubber').value, 10);
    highlightStep();
  };
  refreshSvHint();
}

document.addEventListener('DOMContentLoaded', bindUI);
