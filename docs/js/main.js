// Load results and render the page
let resultsData = null;

async function loadResults() {
  try {
    const resp = await fetch('data/results.json');
    resultsData = await resp.json();
    renderCharts();
    renderResultsTable();
    renderGallery();
    fillAutoSAMRow();
  } catch (e) {
    console.error('Failed to load results:', e);
  }
}

function fillDynamicRow(prefix, summaryKey) {
  const s = resultsData?.summary?.[summaryKey];
  if (!s) return;
  const baseline = resultsData?.summary?.generic_texture;
  const iouEl = document.getElementById(`${prefix}-iou`);
  const ariEl = document.getElementById(`${prefix}-ari`);
  const dIouEl = document.getElementById(`${prefix}-delta-iou`);
  const dAriEl = document.getElementById(`${prefix}-delta-ari`);
  if (iouEl) iouEl.innerHTML = `<strong>${(s.mean_iou).toFixed(3)}</strong>`;
  if (ariEl) ariEl.textContent = (s.mean_ari).toFixed(3);
  if (baseline && dIouEl) {
    const d = s.mean_iou - baseline.mean_iou;
    const cls = d >= 0 ? 'good' : 'bad';
    dIouEl.innerHTML = `<span class="delta-pill ${cls}">${d >= 0 ? '+' : ''}${d.toFixed(3)}</span>`;
  }
  if (baseline && dAriEl) {
    const d = s.mean_ari - baseline.mean_ari;
    const cls = d >= 0 ? 'good' : 'bad';
    dAriEl.innerHTML = `<span class="delta-pill ${cls}">${d >= 0 ? '+' : ''}${d.toFixed(3)}</span>`;
  }
}

function fillAutoSAMRow() {
  fillDynamicRow('autosam', 'autosam');
  fillDynamicRow('sa2va', 'sa2va');
}

// Approach display metadata
const APPROACHES = {
  generic_texture:      { label: 'Generic ZS',       short: 'Generic',     color: '#8b949e' },
  oracle_text:          { label: 'Oracle Text',       short: 'OracleTxt',   color: '#3fb950' },
  points_only:          { label: 'Points Only',       short: 'PtsOnly',     color: '#a5d6a7' },
  oracle_text_points:   { label: 'Oracle Text+Pts',   short: 'Orc T+P',    color: '#1f6feb' },
  qwen3_text_only:      { label: 'Qw3 Proposal',      short: 'Qw3Prop',    color: '#d29922' },
  qwen3_text_proposal:  { label: 'Qw3 Proposal',      short: 'Qw3Prop',    color: '#d29922' },
  qwen3_text_semseg:    { label: 'Qw3 SemSeg',        short: 'Qw3Sem',     color: '#f0883e' },
  qwen3_text_points:    { label: 'Qw3 Text+Pts',      short: 'Qw3T+P',     color: '#bc8cff' },
  qwen3_clipseg:        { label: 'Qw3+CLIPSeg',       short: 'Qw3+CSeg',   color: '#f85149' },
  qwen3_clipseg_sem:    { label: 'Qw3+CSeg(sem)',     short: 'CSeg(sem)',   color: '#da3633' },
  sa2va:                { label: 'Sa2VA',              short: 'Sa2VA',      color: '#e17055' },
  autosam:              { label: 'AutoSAM',           short: 'AutoSAM',    color: '#6c5ce7' },
};

function renderCharts() {
  const summary = resultsData.summary;
  const ctx = document.getElementById('iouChart');
  if (!ctx || !window.Chart) return;

  // Filter out approaches that exist in summary
  const keys = Object.keys(summary).filter(k => typeof summary[k] === 'object' && summary[k].mean_iou != null);

  const labels = keys.map(k => (APPROACHES[k] || {}).short || k);
  const colors = keys.map(k => (APPROACHES[k] || {}).color || '#58a6ff');

  // mIoU chart
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Mean IoU',
          data: keys.map(k => (summary[k].mean_iou * 100).toFixed(1)),
          backgroundColor: colors.map(c => c + '99'),
          borderColor: colors,
          borderWidth: 2,
        },
        {
          label: 'Mean ARI',
          data: keys.map(k => (summary[k].mean_ari * 100).toFixed(1)),
          backgroundColor: colors.map(c => c + '44'),
          borderColor: colors.map(c => c + 'aa'),
          borderWidth: 2,
          borderDash: [5, 3],
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--ink').trim() || '#122033' } },
        title: {
          display: true,
          text: 'mIoU & mARI Comparison (253 samples)',
          color: getComputedStyle(document.documentElement).getPropertyValue('--ink').trim(),
          font: { size: 16 },
        },
      },
      scales: {
        y: {
          min: 50,
          max: 100,
          ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--muted').trim(), callback: v => v + '%' },
          grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--line').trim() },
        },
        x: {
          ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() },
          grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--line').trim() },
        },
      },
    },
  });

  // ARI chart
  const ctx2 = document.getElementById('ariChart');
  if (ctx2) {
    new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Mean ARI',
          data: keys.map(k => (summary[k].mean_ari * 100).toFixed(1)),
          backgroundColor: colors.map(c => c + '77'),
          borderColor: colors,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: getComputedStyle(document.documentElement).getPropertyValue('--ink').trim() } },
          title: {
            display: true,
            text: 'Adjusted Rand Index',
            color: getComputedStyle(document.documentElement).getPropertyValue('--ink').trim(),
            font: { size: 16 },
          },
        },
        scales: {
          y: {
            min: 50,
            max: 100,
            ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--muted').trim(), callback: v => v + '%' },
            grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--line').trim() },
          },
          x: {
            ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() },
            grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--line').trim() },
          },
        },
      },
    });
  }
}

function renderResultsTable() {
  const summary = resultsData.summary;
  const tbody = document.getElementById('results-tbody');
  if (!tbody) return;

  const keys = Object.keys(summary).filter(k => typeof summary[k] === 'object' && summary[k].mean_iou != null);
  const metrics = ['mean_iou', 'mean_dice', 'mean_ari'];

  // Find best values per metric
  const best = {};
  metrics.forEach(m => {
    best[m] = Math.max(...keys.map(k => summary[k][m] || 0));
  });

  tbody.innerHTML = keys.map(k => {
    const d = summary[k];
    const info = APPROACHES[k] || { label: k };
    return `<tr>
      <td>${info.label}</td>
      <td class="${d.mean_iou >= best.mean_iou - 0.001 ? 'best' : ''}">${(d.mean_iou * 100).toFixed(2)}%</td>
      <td class="${d.mean_dice >= best.mean_dice - 0.001 ? 'best' : ''}">${(d.mean_dice * 100).toFixed(2)}%</td>
      <td class="${d.mean_ari >= best.mean_ari - 0.001 ? 'best' : ''}">${(d.mean_ari * 100).toFixed(2)}%</td>
      <td>${d.num_samples}</td>
    </tr>`;
  }).join('');
}

// Gallery
let currentSort = 'name';
let currentFilter = '';

function renderGallery() {
  const grid = document.getElementById('gallery-grid');
  if (!grid || !resultsData.samples) return;

  let samples = [...resultsData.samples];

  // Filter
  if (currentFilter) {
    const f = currentFilter.toLowerCase();
    samples = samples.filter(s =>
      s.crop_name.toLowerCase().includes(f) ||
      (s.descriptions.desc_a || '').toLowerCase().includes(f) ||
      (s.descriptions.desc_b || '').toLowerCase().includes(f)
    );
  }

  // Sort
  if (currentSort === 'iou-asc') {
    samples.sort((a, b) => {
      const aIou = a.metrics?.qwen3_text_proposal?.mean_iou || a.metrics?.qwen3_text_only?.mean_iou || 0;
      const bIou = b.metrics?.qwen3_text_proposal?.mean_iou || b.metrics?.qwen3_text_only?.mean_iou || 0;
      return aIou - bIou;
    });
  } else if (currentSort === 'iou-desc') {
    samples.sort((a, b) => {
      const aIou = a.metrics?.qwen3_text_proposal?.mean_iou || a.metrics?.qwen3_text_only?.mean_iou || 0;
      const bIou = b.metrics?.qwen3_text_proposal?.mean_iou || b.metrics?.qwen3_text_only?.mean_iou || 0;
      return bIou - aIou;
    });
  } else {
    samples.sort((a, b) => a.crop_name.localeCompare(b.crop_name, undefined, { numeric: true }));
  }

  grid.innerHTML = samples.map(s => {
    const iou = s.metrics?.qwen3_text_proposal?.mean_iou || s.metrics?.qwen3_text_only?.mean_iou || 0;
    const iouClass = iou >= 0.85 ? 'iou-high' : iou >= 0.7 ? 'iou-mid' : 'iou-low';
    const fullPath = `assets/thumbnails/${s.crop_name}_oracle_points.jpg`;
    const id = `sample-${s.crop_name}`;

    // Build per-approach metrics summary
    const metricKeys = ['generic', 'oracle_text', 'oracle_text_points',
                        'qwen3_text_proposal', 'qwen3_text_semseg', 'sa2va', 'autosam'];
    const metricLabels = ['Generic', 'OracleTxt', 'Orc T+P', 'Qw3Prop', 'Qw3Sem', 'Sa2VA', 'AutoSAM'];
    let metricsHtml = metricKeys.map((k, i) => {
      const m = s.metrics?.[k] || s.metrics?.[k.replace('_proposal','_only')];
      if (!m) return '';
      const v = m.mean_iou || 0;
      const cls = v >= 0.85 ? 'iou-high' : v >= 0.7 ? 'iou-mid' : 'iou-low';
      return `<span class="iou-badge ${cls}" style="margin:2px">${metricLabels[i]}: ${(v*100).toFixed(1)}%</span>`;
    }).join('');

    // 4 separate figure paths
    const figSuffixes = ['01_gt', '02_baseline', '03_oracle', '04_qwen'];
    const figLabels = ['Ground Truth & Points', 'Baseline (Generic ZS)', 'Oracle Approaches', 'Qwen3-VL (Proposal vs SemSeg)'];

    return `<div class="gallery-item" id="${id}">
      <div class="gallery-header" onclick="toggleExpand('${id}', '${s.crop_name}')">
        <div style="padding:20px">
          <span class="crop-name">${s.crop_name}</span>
          <span class="iou-badge ${iouClass}" style="margin-left:8px">Best IoU: ${(iou * 100).toFixed(1)}%</span>
          <span class="expand-icon" style="float:right;color:var(--muted);font-size:1.2rem">&#9660;</span>
          ${s.descriptions.desc_a ? `<div style="font-size:0.85rem;color:var(--bad);margin-top:8px">A: ${s.descriptions.desc_a}</div>` : ''}
          ${s.descriptions.desc_b ? `<div style="font-size:0.85rem;color:var(--brand)">B: ${s.descriptions.desc_b}</div>` : ''}
          <div style="margin-top:8px">${metricsHtml}</div>
        </div>
      </div>
      <div class="gallery-expand" style="display:none">
        ${figSuffixes.map((suf, i) => `
          <div style="padding:8px 16px 4px; color:var(--muted); font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">${figLabels[i]}</div>
          <img data-suffix="${suf}" alt="${s.crop_name} ${figLabels[i]}" loading="lazy" style="width:100%; display:block; border-bottom:1px solid var(--line);">
        `).join('')}
      </div>
    </div>`;
  }).join('');
}

// Expand/collapse gallery items
function toggleExpand(id, cropName) {
  const item = document.getElementById(id);
  const expandDiv = item.querySelector('.gallery-expand');
  const icon = item.querySelector('.expand-icon');
  const imgs = expandDiv.querySelectorAll('img[data-suffix]');

  if (expandDiv.style.display === 'none') {
    // Load images on first expand
    imgs.forEach(img => {
      if (!img.src || img.src === window.location.href) {
        img.src = `assets/thumbnails/${cropName}_${img.dataset.suffix}.jpg`;
      }
    });
    expandDiv.style.display = 'block';
    icon.innerHTML = '&#9650;';
    item.style.borderColor = 'var(--accent)';
  } else {
    expandDiv.style.display = 'none';
    icon.innerHTML = '&#9660;';
    item.style.borderColor = '';
  }
}

function closeLightbox() {
  const lb = document.getElementById('lightbox');
  if (lb) lb.classList.remove('active');
}

// Events
document.addEventListener('DOMContentLoaded', () => {
  loadResults();

  const search = document.getElementById('gallery-search');
  if (search) {
    search.addEventListener('input', e => {
      currentFilter = e.target.value;
      renderGallery();
    });
  }

  const sort = document.getElementById('gallery-sort');
  if (sort) {
    sort.addEventListener('change', e => {
      currentSort = e.target.value;
      renderGallery();
    });
  }

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeLightbox();
  });
});

// Theme toggle
function toggleTheme() {
  const html = document.documentElement;
  const current = html.dataset.theme || 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  html.dataset.theme = next;
  html.style.colorScheme = next;
  try { localStorage.setItem('sam3-rwtd-theme', next); } catch(_) {}
}
