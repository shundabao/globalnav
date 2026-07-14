const MODE_COLORS = {
  walk: '#36a853',
  drive: '#1f6ed4',
  bus: '#e3731f',
  train: '#e3731f',
  tram: '#e3731f',
  fly: '#6a45c2',
  maritime: '#0f91a8',
};

const MODE_ICONS = {
  walk: 'footprints',
  drive: 'car-front',
  bus: 'bus-front',
  train: 'train-front',
  tram: 'train-front',
  fly: 'plane',
  maritime: 'ship',
};

const ACTION_ICONS = {
  forward: 'arrow-up',
  left: 'corner-up-left',
  right: 'corner-up-right',
  turn_around: 'rotate-ccw',
  u_turn: 'undo-2',
  stop: 'octagon',
  board: 'users-round',
  takeoff: 'plane-takeoff',
  cruise: 'move-right',
  land: 'plane-landing',
  taxi: 'car-front',
  arrive: 'map-pin',
  depart: 'ship',
  dock: 'anchor',
};

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

const TRANSIT_MODES = new Set(['bus', 'train', 'tram']);

const state = {
  tripPlan: null,
  selections: {},
  activeSegmentId: null,
  openSegmentIds: new Set(),
  preferredModes: new Set(['walk', 'drive', 'fly', 'maritime', 'transit']),
  pendingClarify: null,
  map: null,
  routeLayers: [],
  marker: null,
  playbackSessionId: null,
  playbackReady: false,
  trajectory: [],
  stepIndex: -1,
  playing: false,
  playTimer: null,
  playbackSpeed: 1,
  currentObservation: null,
};

const $ = (id) => document.getElementById(id);

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function formatDuration(min) {
  if (!min) return '--:--';
  if (min < 1) return '< 1 min';
  if (min < 60) return `${Math.max(1, Math.round(min))} min`;
  const h = min / 60;
  if (h < 24) return `${h.toFixed(1)} h`;
  return `${Math.floor(h / 24)}d ${Math.round(h % 24)}h`;
}

function modeColor(mode) {
  return MODE_COLORS[mode] || '#94a3b8';
}

function modeIcon(mode) {
  return MODE_ICONS[mode] || 'route';
}

function actionLabel(action) {
  return ACTION_LABELS[action] || String(action || '').replace(/_/g, ' ');
}

function icon(name, className = '') {
  return `<i ${className ? `class="${className}" ` : ''}data-lucide="${name}"></i>`;
}

function selectedOptionFor(seg) {
  const oid = state.selections[seg.segment_id] || seg.default_option_id;
  return seg.options.find((o) => o.option_id === oid) || seg.options[0];
}

function optionModes(opt) {
  return new Set((opt?.micro_legs || []).map((m) => m.mode));
}

function modeMatchesPreference(mode) {
  return state.preferredModes.has(mode) || (TRANSIT_MODES.has(mode) && state.preferredModes.has('transit'));
}

function optionMatchesPreferences(opt) {
  const modes = optionModes(opt);
  if (!modes.size || !state.preferredModes.size) return true;
  return Array.from(modes).some(modeMatchesPreference);
}

function applyPreferencesToSelections() {
  if (!state.tripPlan) return;
  state.tripPlan.segments.forEach((seg) => {
    const current = selectedOptionFor(seg);
    if (current && optionMatchesPreferences(current)) return;
    const preferred = seg.options.find(optionMatchesPreferences);
    state.selections[seg.segment_id] = (preferred || current || seg.options[0])?.option_id || seg.default_option_id;
  });
}

function applyDemoSelectionPreset() {
  if (!state.tripPlan) return;
  const isDefaultDemo = /new york/i.test(state.tripPlan.origin || '')
    && /heathrow|lhr/i.test(state.tripPlan.destination || '');
  if (!isDefaultDemo) return;
  const desiredModes = ['walk', 'fly', 'drive'];
  state.tripPlan.segments.forEach((seg, index) => {
    const desired = desiredModes[index];
    const preferred = seg.options.find((opt) => {
      const modes = optionModes(opt);
      return modes.size === 1 && modes.has(desired);
    });
    if (preferred) state.selections[seg.segment_id] = preferred.option_id;
  });
}

function initMap() {
  if (state.map || typeof L === 'undefined') return;
  state.map = L.map('map', { zoomControl: true }).setView([45, -25], 3);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 19,
  }).addTo(state.map);
  state.marker = L.circleMarker([0, 0], {
    radius: 8,
    color: '#fff',
    weight: 2,
    fillColor: '#1f6ed4',
    fillOpacity: 0.95,
  });
}

function clearRouteLayers() {
  if (!state.map) return;
  state.routeLayers.forEach((layer) => state.map.removeLayer(layer));
  state.routeLayers = [];
}

function geometryFromSelections() {
  if (!state.tripPlan) return { segments: [] };
  return {
    segments: state.tripPlan.segments.map((seg) => {
      const opt = selectedOptionFor(seg);
      return {
        segment_id: seg.segment_id,
        title: seg.title,
        polylines: (opt?.micro_legs || [])
          .map((m) => ({
            mode: m.mode,
            from: m.from,
            to: m.to,
            geometry: m.geometry || [],
          }))
          .filter((pl) => pl.geometry.length),
      };
    }),
  };
}

function drawRouteGeometry(geo, focusSegmentId = null) {
  initMap();
  if (!state.map) return;
  clearRouteLayers();
  const allBounds = [];
  const focusBounds = [];
  (geo?.segments || []).forEach((seg) => {
    (seg.polylines || []).forEach((pl) => {
      const latlngs = (pl.geometry || []).map((p) => [p[0], p[1]]);
      if (!latlngs.length) return;
      const layer = L.polyline(latlngs, {
        color: modeColor(pl.mode),
        weight: pl.mode === 'fly' ? 3.5 : 4.5,
        opacity: 0.88,
        dashArray: pl.mode === 'fly' ? '8 9' : null,
      }).addTo(state.map);
      layer.bindTooltip(`${pl.mode}: ${pl.from} → ${pl.to}`);
      state.routeLayers.push(layer);
      allBounds.push(...latlngs);
      if (focusSegmentId && seg.segment_id === focusSegmentId) focusBounds.push(...latlngs);
    });
  });
  const targetBounds = focusBounds.length ? focusBounds : allBounds;
  if (targetBounds.length) {
    const fitOptions = { padding: [42, 42] };
    if (focusBounds.length) fitOptions.maxZoom = 13;
    state.map.fitBounds(targetBounds, fitOptions);
  }
  setTimeout(() => state.map.invalidateSize(), 80);
}

function updateMapFromSelections(focusSegmentId = null) {
  drawRouteGeometry(geometryFromSelections(), focusSegmentId);
}

function computeTotals() {
  let totalMin = 0;
  let totalKm = 0;
  let selectedCount = 0;
  (state.tripPlan?.segments || []).forEach((seg) => {
    const opt = selectedOptionFor(seg);
    if (!opt) return;
    totalMin += Number(opt.duration_min || 0);
    totalKm += Number(opt.distance_km || 0);
    selectedCount += 1;
  });
  return { totalMin, totalKm, selectedCount };
}

function updateSummary() {
  const plan = state.tripPlan;
  $('originValue').textContent = plan?.origin || 'City Hall, New York';
  $('destinationValue').textContent = plan?.destination || 'University College London';
  const { totalMin, totalKm, selectedCount } = computeTotals();
  $('totalTime').textContent = selectedCount ? formatDuration(totalMin) : '--:--';
  $('totalKm').textContent = selectedCount ? `${totalKm.toFixed(1)} km` : '-- --';
  $('segmentHint').textContent = selectedCount ? `${selectedCount} selected` : 'No route yet';
}

function segmentPrimaryMode(opt) {
  return (opt?.micro_legs || [])[0]?.mode || 'route';
}

function segmentTitle(seg, index, opt) {
  const mode = segmentPrimaryMode(opt);
  const label = opt?.label || seg.segment_type || 'Route';
  return `Segment ${index + 1}: ${label}`;
}

function renderSegments() {
  const root = $('segments');
  if (!state.tripPlan?.segments?.length) {
    root.className = 'segment-list empty';
    root.innerHTML = 'Plan a route to review collapsible segment options.';
    return;
  }

  root.className = 'segment-list';
  root.innerHTML = state.tripPlan.segments.map((seg, index) => {
    const opt = selectedOptionFor(seg);
    const primaryMode = segmentPrimaryMode(opt);
    const open = state.openSegmentIds.has(seg.segment_id);
    const active = seg.segment_id === state.activeSegmentId;
    const options = seg.options.map((candidate) => {
      const selected = candidate.option_id === opt?.option_id;
      return `
        <button type="button" class="segment-option ${selected ? 'selected' : ''}" data-seg="${escapeHtml(seg.segment_id)}" data-opt="${escapeHtml(candidate.option_id)}">
          <strong>${escapeHtml(candidate.label)}</strong>
          <small>${escapeHtml(candidate.duration_display)} · ${escapeHtml(candidate.distance_km)} km · ${escapeHtml(candidate.mode_chain || '')}</small>
        </button>`;
    }).join('');

    return `
      <article class="route-segment ${open ? 'open' : ''} ${active ? 'active' : ''}" data-seg="${escapeHtml(seg.segment_id)}" style="color:${modeColor(primaryMode)}">
        <span class="segment-stem" aria-hidden="true"></span>
        <button type="button" class="segment-toggle" data-seg="${escapeHtml(seg.segment_id)}" aria-expanded="${open}">
          <span class="segment-icon ${escapeHtml(primaryMode)}">${icon(modeIcon(primaryMode))}</span>
          <span>
            <span class="segment-title-line">${escapeHtml(segmentTitle(seg, index, opt))}</span>
            <span class="segment-subtitle">${escapeHtml(seg.from || '')} → ${escapeHtml(seg.to || '')}</span>
          </span>
          ${icon('chevron-down', 'segment-chevron')}
        </button>
        <div class="segment-body">
          ${options}
        </div>
      </article>`;
  }).join('');

  root.querySelectorAll('.segment-toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      const segId = btn.dataset.seg;
      const shouldOpen = !state.openSegmentIds.has(segId);
      if (shouldOpen) {
        state.openSegmentIds.add(segId);
        state.activeSegmentId = segId;
      } else {
        state.openSegmentIds.delete(segId);
        if (state.activeSegmentId === segId) state.activeSegmentId = null;
      }
      renderSegments();
      updateMapFromSelections(shouldOpen ? segId : null);
      refreshIcons();
    });
  });

  root.querySelectorAll('.segment-option').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.selections[btn.dataset.seg] = btn.dataset.opt;
      state.activeSegmentId = btn.dataset.seg;
      state.openSegmentIds.add(btn.dataset.seg);
      resetPlaybackState('Route changed. Press play to prepare the selected route.');
      updateSummary();
      renderSegments();
      renderTimeline();
      updateMapFromSelections(btn.dataset.seg);
      refreshIcons();
    });
  });

  refreshIcons();
}

function renderTimeline() {
  const root = $('timeline');
  if (!state.tripPlan?.segments?.length) {
    root.className = 'timeline empty';
    root.textContent = 'Plan a route to see the playback timeline.';
    return;
  }
  root.className = 'timeline';
  const steps = state.tripPlan.segments.map((seg, index) => {
    const opt = selectedOptionFor(seg);
    const mode = segmentPrimaryMode(opt);
    const active = state.currentObservation?.segment_id === seg.segment_id || (!state.currentObservation && index === 0);
    return `
      <div class="timeline-step">
        <span class="timeline-dot ${active ? 'active' : ''} timeline-mode ${escapeHtml(mode)}">${icon(modeIcon(mode))}</span>
        <span>Segment ${index + 1}: ${escapeHtml(opt?.label || seg.segment_type)}</span>
      </div>`;
  }).join('');
  root.innerHTML = `<div class="timeline-track">${steps}</div>`;
  refreshIcons();
}

function renderActions(actions = []) {
  const root = $('actionList');
  const list = actions.length ? actions : ['prepare playback'];
  root.innerHTML = list.map((action, index) => {
    if (action === 'prepare playback') {
      return `<button type="button" disabled>${icon('route')}Prepare playback</button>`;
    }
    return `<button type="button" class="${index === 0 ? 'primary-action' : ''}">${icon(ACTION_ICONS[action] || 'circle')}<span>${escapeHtml(actionLabel(action))}</span></button>`;
  }).join('');
  refreshIcons();
}

function showStreetViewPlaceholder(message, detail = '') {
  const frame = document.querySelector('.streetview-frame');
  const img = $('svImage');
  const placeholder = $('svPlaceholder');
  frame.classList.remove('has-image');
  img.removeAttribute('src');
  placeholder.innerHTML = `<span>Street View</span><p>${escapeHtml(message)}</p>${detail ? `<p>${escapeHtml(detail)}</p>` : ''}`;
}

function showStreetViewImage(url, alt) {
  const frame = document.querySelector('.streetview-frame');
  const img = $('svImage');
  const placeholder = $('svPlaceholder');
  frame.classList.remove('has-image');
  placeholder.innerHTML = '<span>Street View</span><p>Loading Street View image...</p>';
  img.onload = () => frame.classList.add('has-image');
  img.onerror = () => showStreetViewPlaceholder('Street View image failed to load', 'You can continue playback without imagery.');
  img.alt = alt || 'Street View';
  img.src = `${url}&_=${Date.now()}`;
}

function updateMarker(obs) {
  if (!obs || !obs.lat || !obs.lon || !state.map) return;
  state.marker.setLatLng([obs.lat, obs.lon]).addTo(state.map);
  state.map.panTo([obs.lat, obs.lon], { animate: true, duration: 0.35 });
}

function renderObservation(obs) {
  state.currentObservation = obs || null;
  $('playbackStatus').textContent = obs ? `${obs.mode || 'route'} · ${obs.progress || 'ready'}` : 'Not prepared';

  const summary = $('obsSummary');
  if (!obs) {
    summary.innerHTML = '';
    $('svDots').innerHTML = '';
    renderActions([]);
    showStreetViewPlaceholder('Prepare playback to show the current route view.');
    return;
  }

  const sv = obs.streetview || {};
  if (sv.available && sv.image_url) {
    showStreetViewImage(sv.image_url, `Street View ${obs.facing || ''}`);
  } else if (sv.reason?.endsWith('_phase')) {
    showStreetViewPlaceholder(`No Street View for the ${obs.mode || 'current'} phase`, `${obs.from || ''} → ${obs.to || ''}`);
  } else {
    showStreetViewPlaceholder(sv.message || 'No Street View available', `${obs.from || ''} → ${obs.to || ''}`);
  }

  summary.innerHTML = `
    <div class="obs-row"><span>Mode</span><span>${escapeHtml(obs.mode || '-')}</span></div>
    <div class="obs-row"><span>Facing</span><span>${escapeHtml(obs.facing || '?')}${obs.heading_deg != null ? ` · ${escapeHtml(obs.heading_deg)}°` : ''}</span></div>
    <div class="obs-row"><span>Route</span><span>${escapeHtml(obs.from || '-')} → ${escapeHtml(obs.to || '-')}</span></div>
    <div class="obs-row"><span>Progress</span><span>${escapeHtml(obs.progress || '-')}</span></div>`;

  const dotCount = Math.max(6, Math.min(8, state.tripPlan?.segments?.length ? state.tripPlan.segments.length * 2 : 6));
  const activeDot = Math.max(0, Math.min(dotCount - 1, state.stepIndex));
  $('svDots').innerHTML = Array.from({ length: dotCount }, (_, i) => `<span class="sv-dot ${i === activeDot ? 'active' : ''}"></span>`).join('');

  renderActions(obs.available_actions || []);
  updateMarker(obs);
  renderTimeline();
}

function resetPlaybackState(message = 'Press play to prepare the selected route.') {
  stopPlayback();
  state.playbackSessionId = null;
  state.playbackReady = false;
  state.trajectory = [];
  state.stepIndex = -1;
  state.currentObservation = null;
  $('playBtn').innerHTML = icon('play');
  renderObservation(null);
  $('playbackStatus').textContent = message;
  renderTimeline();
  refreshIcons();
}

async function ensurePlaybackSession() {
  if (!state.tripPlan) {
    $('status').textContent = 'Plan a route before starting playback.';
    return false;
  }
  if (state.playbackReady && state.playbackSessionId) return true;

  $('playbackStatus').textContent = 'Preparing...';
  $('status').textContent = 'Preparing route playback for the selected options...';
  const res = await fetch('/api/follower/prepare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      instruction: $('instruction').value.trim(),
      origin: state.tripPlan.origin,
      destination: state.tripPlan.destination,
      selections: state.selections,
      rule_based: true,
      no_llm: true,
      streetview: true,
    }),
  });
  const data = await res.json();
  if (data.error) {
    $('status').textContent = `Playback error: ${data.error}`;
    $('playbackStatus').textContent = 'Playback unavailable';
    return false;
  }
  if (data.status === 'clarifying') {
    showClarify(data.questions);
    return false;
  }

  state.playbackSessionId = data.session_id;
  state.playbackReady = true;
  state.trajectory = [];
  state.stepIndex = -1;
  state.currentObservation = data.initial_observation;
  $('status').textContent = 'Playback is ready.';
  drawRouteGeometry(data.route_geometry || geometryFromSelections());
  renderObservation(data.initial_observation);
  return true;
}

async function stepOnce() {
  const ready = await ensurePlaybackSession();
  if (!ready || !state.playbackSessionId) return false;
  const res = await fetch('/api/follower/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: state.playbackSessionId }),
  });
  const data = await res.json();
  if (data.error) {
    $('status').textContent = `Playback error: ${data.error}`;
    stopPlayback();
    return false;
  }
  state.trajectory.push(data);
  state.stepIndex = state.trajectory.length - 1;
  renderObservation(data.observation || data.observation_after || state.currentObservation);
  $('status').textContent = data.done ? `Playback complete: success=${data.success}` : `Step ${data.step}: ${actionLabel(data.action)}`;
  if (data.done) {
    stopPlayback();
    return false;
  }
  return true;
}

function stopPlayback() {
  state.playing = false;
  if (state.playTimer) clearInterval(state.playTimer);
  state.playTimer = null;
  $('playBtn').innerHTML = icon('play');
  refreshIcons();
}

async function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    return;
  }
  const ready = await ensurePlaybackSession();
  if (!ready) return;
  state.playing = true;
  $('playBtn').innerHTML = icon('pause');
  refreshIcons();
  await stepOnce();
  state.playTimer = setInterval(async () => {
    const keepGoing = await stepOnce();
    if (!keepGoing) stopPlayback();
  }, Math.max(300, 900 / state.playbackSpeed));
}

function buildPlanPayload() {
  const instruction = $('instruction').value.trim();
  const match = instruction.match(/\bfrom\s+(.+?)\s+to\s+(.+?)(?:[.!?]|$)/i);
  return {
    instruction,
    origin: match?.[1]?.trim(),
    destination: match?.[2]?.trim(),
  };
}

async function planTrip() {
  const payload = buildPlanPayload();
  if (!payload.instruction) return;
  $('planBtn').disabled = true;
  $('status').textContent = 'Generating a multi-modal route...';
  resetPlaybackState('Waiting for route');

  try {
    const res = await fetch('/api/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    handlePlanResponse(data);
  } catch (e) {
    $('status').textContent = `Request failed: ${e.message}`;
  } finally {
    $('planBtn').disabled = false;
  }
}

function handlePlanResponse(data) {
  if (data.status === 'clarifying') {
    showClarify(data.questions);
    return;
  }
  if (data.error) {
    $('status').textContent = `Error: ${data.error}`;
    return;
  }
  state.tripPlan = data;
  state.selections = {};
  state.activeSegmentId = null;
  state.openSegmentIds = new Set();
  (data.segments || []).forEach((seg) => {
    state.selections[seg.segment_id] = seg.default_option_id;
  });
  if (data.segments?.[0]?.segment_id) {
    state.activeSegmentId = data.segments[0].segment_id;
    state.openSegmentIds.add(data.segments[0].segment_id);
  }
  applyDemoSelectionPreset();
  applyPreferencesToSelections();
  updateSummary();
  renderSegments();
  renderTimeline();
  drawRouteGeometry(geometryFromSelections());
  renderObservation(null);
  $('status').textContent = data.warning || 'Route generated. Expand each segment to inspect and choose route options.';
  refreshIcons();
}

function showClarify(questions) {
  state.pendingClarify = questions[0];
  $('clarifyQuestion').textContent = state.pendingClarify.question;
  const optsEl = $('clarifyOptions');
  optsEl.innerHTML = '';
  $('clarifyInput').classList.add('hidden');

  if (state.pendingClarify.options && state.pendingClarify.options.length) {
    state.pendingClarify.options.forEach((opt) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'clarify-opt';
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
  const res = await fetch('/api/clarify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      instruction: $('instruction').value.trim(),
      category: state.pendingClarify.category,
      question: state.pendingClarify.question,
      answer: answer || $('clarifyInput').value.trim(),
    }),
  });
  const data = await res.json();
  handlePlanResponse(data);
}

function bindModePreferences() {
  document.querySelectorAll('.mode-pill').forEach((button) => {
    button.addEventListener('click', () => {
      const mode = button.dataset.mode;
      if (state.preferredModes.has(mode)) {
        state.preferredModes.delete(mode);
      } else {
        state.preferredModes.add(mode);
      }
      button.classList.toggle('active', state.preferredModes.has(mode));
      if (state.tripPlan) {
        applyPreferencesToSelections();
        resetPlaybackState('Route preferences changed. Press play to prepare the selected route.');
        updateSummary();
        renderSegments();
        renderTimeline();
        updateMapFromSelections(state.activeSegmentId);
      }
    });
  });
}

function bindUI() {
  initMap();
  refreshIcons();
  bindModePreferences();
  renderObservation(null);
  renderTimeline();
  updateSummary();

  $('planBtn').addEventListener('click', planTrip);
  $('instruction').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') planTrip();
  });
  $('playBtn').addEventListener('click', togglePlayback);
  $('resetPlaybackBtn').addEventListener('click', () => resetPlaybackState());
  $('fitBtn').addEventListener('click', () => updateMapFromSelections());
  document.querySelector('.map-tool')?.addEventListener('click', () => updateMapFromSelections(state.activeSegmentId));
  $('speedBtn').addEventListener('click', () => {
    const speeds = [1, 1.5, 2];
    const next = speeds[(speeds.indexOf(state.playbackSpeed) + 1) % speeds.length];
    state.playbackSpeed = next;
    $('speedBtn').textContent = `${next}x`;
    if (state.playing) {
      stopPlayback();
      togglePlayback();
    }
  });
  $('swapBtn').addEventListener('click', () => {
    if (!state.tripPlan) return;
    $('instruction').value = `Plan a route from ${state.tripPlan.destination} to ${state.tripPlan.origin}.`;
    planTrip();
  });
  $('clarifySubmit').addEventListener('click', () => submitClarify());
  setTimeout(planTrip, 250);
}

document.addEventListener('DOMContentLoaded', bindUI);
