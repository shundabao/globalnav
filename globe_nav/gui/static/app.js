let tripPlan = null;
let selections = {};
let pendingClarify = null;

const $ = (id) => document.getElementById(id);

function formatDuration(min) {
  if (min < 1) return '< 1 min';
  if (min < 60) return `${Math.max(1, Math.round(min))} min`;
  const h = min / 60;
  if (h < 24) return `${h.toFixed(1)} h`;
  return `${Math.floor(h / 24)}d ${Math.round(h % 24)}h`;
}

function updateTotal() {
  if (!tripPlan) return;
  let totalMin = 0;
  let totalKm = 0;
  for (const seg of tripPlan.segments) {
    const oid = selections[seg.segment_id] || seg.default_option_id;
    const opt = seg.options.find((o) => o.option_id === oid) || seg.options[0];
    if (opt) {
      totalMin += opt.duration_min;
      totalKm += opt.distance_km;
    }
  }
  $('totalTime').textContent = formatDuration(totalMin);
  $('totalKm').textContent = `${totalKm.toFixed(0)} km`;
}

function renderMicroLegs(segId, opt) {
  const el = document.getElementById(`micro-${segId}`);
  if (!el) return;
  el.innerHTML = opt.micro_legs
    .map(
      (m, i) => `
    <div class="micro-leg">
      <div class="micro-leg-title">${i + 1}. [${m.mode}] ${m.from} → ${m.to}
        <span style="color:var(--muted)">(${m.duration_display}, ${m.distance_km} km)</span>
      </div>
      ${m.steps.slice(0, 6).map((s) => `<div class="micro-step">→ ${s}</div>`).join('')}
      ${m.note ? `<div class="micro-step">${m.note}</div>` : ''}
    </div>`
    )
    .join('');
  el.classList.add('visible');
}

function renderSegments() {
  const container = $('segments');
  container.innerHTML = '';

  tripPlan.segments.forEach((seg, idx) => {
    const card = document.createElement('div');
    card.className = 'segment-card';

    const typeBadge =
      seg.segment_type === 'flight'
        ? '<span class="badge flight">航班</span>'
        : '<span class="badge local">本地</span>';

    card.innerHTML = `
      <div class="segment-header">
        <span class="segment-num">${idx + 1}</span>
        <span class="segment-title">${seg.title}</span>
        ${typeBadge}
      </div>
      <div class="segment-desc">${seg.description || ''}</div>
      <div class="options-grid" id="opts-${seg.segment_id}"></div>
      <div class="micro-legs" id="micro-${seg.segment_id}"></div>
    `;
    container.appendChild(card);

    const grid = card.querySelector(`#opts-${seg.segment_id}`);
    const defaultId = seg.default_option_id;
    selections[seg.segment_id] = selections[seg.segment_id] || defaultId;

    seg.options.forEach((opt) => {
      const chip = document.createElement('div');
      chip.className = 'option-chip';
      if (opt.option_id === selections[seg.segment_id]) chip.classList.add('selected');
      if (opt.is_recommended) chip.classList.add('recommended');

      chip.innerHTML = `
        <div class="tooltip">${opt.tooltip || opt.duration_display}</div>
        <div class="option-label">${opt.label}</div>
        <div class="option-meta">${opt.duration_display} · ${opt.distance_km} km</div>
      `;

      chip.addEventListener('mouseenter', () => {
        chip.querySelector('.tooltip').textContent =
          opt.tooltip || `${opt.duration_display} · ${opt.distance_km} km`;
      });

      chip.addEventListener('click', () => {
        selections[seg.segment_id] = opt.option_id;
        grid.querySelectorAll('.option-chip').forEach((c) => c.classList.remove('selected'));
        chip.classList.add('selected');
        renderMicroLegs(seg.segment_id, opt);
        updateTotal();
      });

      grid.appendChild(chip);
    });

    const selectedOpt =
      seg.options.find((o) => o.option_id === selections[seg.segment_id]) || seg.options[0];
    if (selectedOpt) renderMicroLegs(seg.segment_id, selectedOpt);
  });

  updateTotal();
}

function showClarify(questions) {
  pendingClarify = questions[0];
  $('clarifyQuestion').textContent = pendingClarify.question;
  const optsEl = $('clarifyOptions');
  optsEl.innerHTML = '';
  $('clarifyInput').classList.add('hidden');

  if (pendingClarify.options && pendingClarify.options.length) {
    pendingClarify.options.forEach((opt) => {
      const btn = document.createElement('button');
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
      category: pendingClarify.category,
      question: pendingClarify.question,
      answer: answer || $('clarifyInput').value.trim(),
    }),
  });
  const data = await res.json();
  handlePlanResponse(data);
}

function handlePlanResponse(data) {
  if (data.status === 'clarifying') {
    showClarify(data.questions);
    return;
  }
  if (data.error) {
    $('status').textContent = `错误: ${data.error}`;
    return;
  }
  tripPlan = data;
  selections = {};
  $('summary').classList.remove('hidden');
  $('routeLabel').textContent = `${data.origin} → ${data.destination}`;
  $('status').textContent = '点击各段选项切换交通方式，总耗时会实时更新';
  renderSegments();
}

async function planTrip() {
  const instruction = $('instruction').value.trim();
  if (!instruction) return;
  $('planBtn').disabled = true;
  $('status').textContent = 'LLM 解析意图 + 环境生成各段选项…';

  try {
    const res = await fetch('/api/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ instruction }),
    });
    const data = await res.json();
    handlePlanResponse(data);
  } catch (e) {
    $('status').textContent = `请求失败: ${e.message}`;
  } finally {
    $('planBtn').disabled = false;
  }
}

$('planBtn').addEventListener('click', planTrip);
$('instruction').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') planTrip();
});
$('clarifySubmit').addEventListener('click', () => submitClarify());
