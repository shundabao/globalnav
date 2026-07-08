const ACTIONS_BY_MODE = {
  walk: ['forward', 'left', 'right', 'turn_around', 'stop'],
  drive: ['forward', 'left', 'right', 'u_turn', 'stop'],
  bus: ['board', 'cruise', 'arrive', 'stop'],
  train: ['board', 'cruise', 'arrive', 'stop'],
  tram: ['board', 'cruise', 'arrive', 'stop'],
  fly: ['takeoff', 'cruise', 'land', 'stop'],
  maritime: ['depart', 'cruise', 'dock', 'stop'],
};

const MODE_COLORS = {
  walk: '#0ea5e9',
  drive: '#f59e0b',
  bus: '#16a34a',
  train: '#db2777',
  tram: '#7c3aed',
  fly: '#8b5cf6',
  maritime: '#14b8a6',
};

const state = {
  sessionId: null,
  plan: null,
  selections: {},
  nodes: [],
  currentIndex: -1,
  map: null,
  routeLayers: [],
  nodeLayer: null,
  activeMarker: null,
};

const $ = id => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function modeColor(mode) {
  return MODE_COLORS[mode] || '#94a3b8';
}

function formatDuration(minutes) {
  if (!minutes) return '--';
  if (minutes < 60) return `${Math.max(1, Math.round(minutes))} min`;
  const hours = minutes / 60;
  if (hours < 24) return `${hours.toFixed(1)} h`;
  return `${Math.floor(hours / 24)}d ${Math.round(hours % 24)}h`;
}

function initMap() {
  if (state.map) return;
  state.map = L.map('map', { zoomControl: true }).setView([-33.87, 151.21], 12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 19,
  }).addTo(state.map);
}

function clearRoute() {
  state.routeLayers.forEach(layer => state.map.removeLayer(layer));
  state.routeLayers = [];
}

function drawRouteGeometry(geo) {
  initMap();
  clearRoute();
  const bounds = [];
  (geo?.segments || []).forEach(seg => {
    (seg.polylines || []).forEach(pl => {
      const latlngs = (pl.geometry || []).map(p => [p[0], p[1]]);
      if (!latlngs.length) return;
      const layer = L.polyline(latlngs, {
        color: modeColor(pl.mode),
        weight: pl.mode === 'fly' ? 3 : 4,
        opacity: 0.84,
        dashArray: pl.mode === 'fly' ? '7 8' : null,
      }).addTo(state.map);
      layer.bindTooltip(`${pl.mode}: ${pl.from} → ${pl.to}`);
      state.routeLayers.push(layer);
      bounds.push(...latlngs);
    });
  });
  if (bounds.length) state.map.fitBounds(bounds, { padding: [34, 34] });
  setTimeout(() => state.map.invalidateSize(), 60);
}

function geometryFromPlanSelections() {
  if (!state.plan) return { segments: [] };
  return {
    segments: state.plan.segments.map(seg => {
      const optId = state.selections[seg.segment_id] || seg.default_option_id;
      const opt = seg.options.find(o => o.option_id === optId) || seg.options[0];
      return {
        segment_id: seg.segment_id,
        title: seg.title,
        polylines: (opt?.micro_legs || []).map(m => ({
          mode: m.mode,
          from: m.from,
          to: m.to,
          geometry: m.geometry || [],
        })).filter(pl => pl.geometry.length),
      };
    }),
  };
}

function updateRouteSummary() {
  if (!state.plan) {
    $('routeSummary').textContent = 'No route yet';
    return;
  }
  let totalMin = 0;
  let totalKm = 0;
  state.plan.segments.forEach(seg => {
    const optId = state.selections[seg.segment_id] || seg.default_option_id;
    const opt = seg.options.find(o => o.option_id === optId) || seg.options[0];
    if (!opt) return;
    totalMin += Number(opt.duration_min || 0);
    totalKm += Number(opt.distance_km || 0);
  });
  $('routeSummary').textContent = `${formatDuration(totalMin)} · ${totalKm.toFixed(1)} km`;
}

function preferredAnnotationOption(seg) {
  const groundModes = new Set(['walk', 'drive']);
  return (
    seg.options.find(opt => (opt.micro_legs || []).length && opt.micro_legs.every(m => groundModes.has(m.mode))) ||
    seg.options.find(opt => (opt.micro_legs || []).some(m => groundModes.has(m.mode))) ||
    seg.options.find(opt => opt.option_id === seg.default_option_id) ||
    seg.options[0]
  );
}

function renderSegments() {
  const box = $('segmentList');
  if (!state.plan?.segments?.length) {
    box.className = 'segment-list empty';
    box.textContent = '生成路线后选择每一段的 ground truth option。';
    return;
  }
  box.className = 'segment-list';
  box.innerHTML = state.plan.segments.map(seg => {
    const selected = state.selections[seg.segment_id] || seg.default_option_id;
    const options = seg.options.map(opt => `
      <button type="button" class="option-button ${opt.option_id === selected ? 'active' : ''}"
        data-seg="${escapeHtml(seg.segment_id)}" data-opt="${escapeHtml(opt.option_id)}">
        ${escapeHtml(opt.label)}
        <small>${escapeHtml(opt.duration_display)} · ${escapeHtml(opt.distance_km)} km · ${escapeHtml(opt.mode_chain || '')}</small>
      </button>
    `).join('');
    return `
      <div class="segment-card">
        <div class="segment-title">${escapeHtml(seg.title)}</div>
        <div class="option-grid">${options}</div>
      </div>
    `;
  }).join('');

  box.querySelectorAll('.option-button').forEach(btn => {
    btn.onclick = () => {
      state.selections[btn.dataset.seg] = btn.dataset.opt;
      renderSegments();
      updateRouteSummary();
      drawRouteGeometry(geometryFromPlanSelections());
    };
  });
  updateRouteSummary();
}

function setStreetViewPlaceholder(message, detail = '') {
  const frame = document.querySelector('.streetview-frame');
  const img = $('streetviewImage');
  const placeholder = $('streetviewPlaceholder');
  frame.classList.remove('has-image');
  img.removeAttribute('src');
  placeholder.innerHTML = `<span>Street View</span><p>${escapeHtml(message)}</p>${detail ? `<p>${escapeHtml(detail)}</p>` : ''}`;
}

function showStreetView(node) {
  const frame = document.querySelector('.streetview-frame');
  const img = $('streetviewImage');
  const sv = node?.streetview || {};
  frame.classList.remove('has-image');
  if (sv.available && sv.image_url) {
    $('streetviewPlaceholder').innerHTML = '<span>Street View</span><p>正在加载街景图像…</p>';
    img.onload = () => frame.classList.add('has-image');
    img.onerror = () => setStreetViewPlaceholder('街景图像加载失败', '可以继续标注，或稍后重试这一点');
    img.src = `${sv.image_url}&_=${Date.now()}`;
    return;
  }
  setStreetViewPlaceholder(sv.message || '此节点没有街景', sv.reason || '');
}

function actionOptions(node) {
  const base = new Set(ACTIONS_BY_MODE[node.mode] || ['forward', 'stop']);
  base.add(node.oracle_action);
  return Array.from(base);
}

function persistCurrentNode() {
  const node = state.nodes[state.currentIndex];
  if (!node) return;
  node.annotation_instruction = $('instructionInput').value;
  const chosen = $('actionSelect').value;
  node.action_override = chosen && chosen !== node.oracle_action ? chosen : '';
}

function effectiveAction(node) {
  return node.action_override || node.oracle_action;
}

function renderNodeMeta(node) {
  const action = effectiveAction(node);
  $('nodeMeta').innerHTML = `
    <div class="meta-pill"><span>Mode</span><span>${escapeHtml(node.mode)}</span></div>
    <div class="meta-pill"><span>Oracle</span><span>${escapeHtml(action)}${node.action_override ? ' (override)' : ''}</span></div>
    <div class="meta-pill"><span>Facing</span><span>${escapeHtml(node.facing)} · ${escapeHtml(node.heading_deg)}°</span></div>
    <div class="meta-pill"><span>Location</span><span>${escapeHtml(node.lat)}, ${escapeHtml(node.lon)}</span></div>
    <div class="meta-pill"><span>Route cue</span><span>${escapeHtml(node.route_cue || 'none')}</span></div>
  `;
}

function selectNode(index) {
  if (!state.nodes.length) return;
  if (state.currentIndex >= 0) persistCurrentNode();
  state.currentIndex = Math.max(0, Math.min(index, state.nodes.length - 1));
  const node = state.nodes[state.currentIndex];

  $('nodeCounter').textContent = `${state.currentIndex + 1} / ${state.nodes.length}`;
  $('instructionInput').value = node.annotation_instruction || '';
  $('actionSelect').innerHTML = actionOptions(node).map(action => (
    `<option value="${escapeHtml(action)}">${escapeHtml(action)}${action === node.oracle_action ? ' · auto' : ''}</option>`
  )).join('');
  $('actionSelect').value = effectiveAction(node);
  showStreetView(node);
  renderNodeMeta(node);

  if (state.activeMarker) state.map.removeLayer(state.activeMarker);
  if (node.lat || node.lon) {
    state.activeMarker = L.circleMarker([node.lat, node.lon], {
      radius: 9,
      color: '#fff',
      weight: 2,
      fillColor: '#2563eb',
      fillOpacity: 0.95,
    }).addTo(state.map);
    state.map.panTo([node.lat, node.lon], { animate: true, duration: 0.35 });
  }
  renderNodeList();
  updateProgress();
}

function renderNodeMarkers() {
  initMap();
  if (state.nodeLayer) state.map.removeLayer(state.nodeLayer);
  state.nodeLayer = L.layerGroup().addTo(state.map);
  state.nodes.forEach((node, i) => {
    if (!node.lat && !node.lon) return;
    const marker = L.circleMarker([node.lat, node.lon], {
      radius: node.is_keypoint ? 5 : 3,
      color: node.is_keypoint ? '#d97706' : '#2563eb',
      weight: 1,
      fillOpacity: 0.72,
    }).addTo(state.nodeLayer);
    marker.bindTooltip(`#${i + 1} ${node.oracle_action}`);
    marker.on('click', () => selectNode(i));
  });
}

function renderNodeList() {
  const list = $('nodeList');
  if (!state.nodes.length) {
    list.className = 'node-list empty';
    list.textContent = '开始标注后这里会显示路线节点。';
    return;
  }
  list.className = 'node-list';
  list.innerHTML = state.nodes.map((node, i) => `
    <button type="button"
      class="node-chip ${i === state.currentIndex ? 'current' : ''} ${node.is_keypoint ? 'keypoint' : ''} ${(node.annotation_instruction || '').trim() ? 'annotated' : ''}"
      data-idx="${i}">
      #${i + 1}<br>${escapeHtml(effectiveAction(node))}
    </button>
  `).join('');
  list.querySelectorAll('.node-chip').forEach(btn => {
    btn.onclick = () => selectNode(Number(btn.dataset.idx));
  });
}

function updateProgress() {
  const annotated = state.nodes.filter(n => (n.annotation_instruction || '').trim()).length;
  const keypoints = state.nodes.filter(n => n.is_keypoint).length;
  $('progressText').textContent = `${annotated} annotated · ${keypoints} keypoints · ${state.nodes.length} nodes`;
}

async function refreshStreetViewStatus() {
  try {
    const res = await fetch('/api/streetview/status?enabled=1');
    const data = await res.json();
    $('streetviewState').textContent = data.working ? `${data.source || 'available'}` : (data.message || 'unavailable');
  } catch {
    $('streetviewState').textContent = 'unknown';
  }
}

async function randomPair() {
  $('setupStatus').textContent = '随机选择中…';
  const res = await fetch('/api/annotate/random?scope=local');
  const data = await res.json();
  if (data.error) {
    $('setupStatus').textContent = data.error;
    return;
  }
  $('originInput').value = data.origin;
  $('destinationInput').value = data.destination;
  $('setupStatus').textContent = `${data.city || data.scope}: ${data.origin} → ${data.destination}`;
}

async function planRoute() {
  const origin = $('originInput').value.trim();
  const destination = $('destinationInput').value.trim();
  if (!origin || !destination) return;

  $('planBtn').disabled = true;
  $('startBtn').disabled = true;
  $('setupStatus').textContent = '环境正在计算路线选项…';
  try {
    const res = await fetch('/api/annotate/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ origin, destination }),
    });
    const data = await res.json();
    if (data.error) {
      $('setupStatus').textContent = `错误: ${data.error}`;
      return;
    }
    state.sessionId = data.session_id;
    state.plan = data;
    state.selections = {};
    state.nodes = [];
    state.currentIndex = -1;
    data.segments.forEach(seg => {
      const preferred = preferredAnnotationOption(seg);
      state.selections[seg.segment_id] = preferred?.option_id || seg.default_option_id;
    });
    $('setupStatus').textContent = '路线已生成，选择 option 后固定路线。';
    $('startBtn').disabled = false;
    renderSegments();
    drawRouteGeometry(data.route_geometry);
    renderNodeList();
    updateProgress();
  } catch (e) {
    $('setupStatus').textContent = `请求失败: ${e.message}`;
  } finally {
    $('planBtn').disabled = false;
  }
}

async function startAnnotation() {
  if (!state.sessionId) return;
  $('startBtn').disabled = true;
  $('setupStatus').textContent = '正在生成逐节点标注任务…';
  try {
    const res = await fetch('/api/annotate/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        selections: state.selections,
        sample_every: Number($('sampleEveryInput').value || 1),
        ground_only: $('groundOnlyInput').checked,
      }),
    });
    const data = await res.json();
    if (data.error) {
      $('setupStatus').textContent = `错误: ${data.error}`;
      return;
    }
    state.nodes = data.nodes || [];
    state.currentIndex = -1;
    $('setupStatus').textContent = `已固定路线：${data.node_count} 个节点。`;
    drawRouteGeometry(data.route_geometry);
    renderNodeMarkers();
    renderNodeList();
    updateProgress();
    if (state.nodes.length) selectNode(0);
  } catch (e) {
    $('setupStatus').textContent = `请求失败: ${e.message}`;
  } finally {
    $('startBtn').disabled = false;
  }
}

async function saveAnnotations() {
  if (!state.nodes.length) return;
  persistCurrentNode();
  $('saveStatus').textContent = '保存中…';
  try {
    const res = await fetch('/api/annotate/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        annotator: $('annotatorInput').value.trim(),
        selections: state.selections,
        nodes: state.nodes,
      }),
    });
    const data = await res.json();
    if (data.error) {
      $('saveStatus').textContent = `保存失败: ${data.error}`;
      return;
    }
    $('saveStatus').textContent = `已保存 ${data.record_id} · ${data.node_count} nodes`;
  } catch (e) {
    $('saveStatus').textContent = `保存失败: ${e.message}`;
  }
}

function bindUI() {
  initMap();
  refreshStreetViewStatus();
  $('randomBtn').onclick = randomPair;
  $('planBtn').onclick = planRoute;
  $('startBtn').onclick = startAnnotation;
  $('prevNodeBtn').onclick = () => selectNode(state.currentIndex - 1);
  $('nextNodeBtn').onclick = () => selectNode(state.currentIndex + 1);
  $('saveBtn').onclick = saveAnnotations;
  $('instructionInput').addEventListener('input', () => {
    persistCurrentNode();
    renderNodeList();
    updateProgress();
  });
  $('actionSelect').addEventListener('change', () => {
    persistCurrentNode();
    renderNodeMeta(state.nodes[state.currentIndex]);
    renderNodeList();
  });
}

document.addEventListener('DOMContentLoaded', bindUI);
