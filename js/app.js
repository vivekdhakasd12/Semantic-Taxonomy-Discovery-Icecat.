const detailsPanel  = document.getElementById('details-panel');
const detailsContent = document.getElementById('details-content');
let rootData    = null;
let clusterData = [];
let richData    = {};
let groupedData = {};
let categoryMode        = 'top30';
let TOP_N               = 30;
let currentPurity       = 0.70; 
const INITIAL_CLUSTERS  = 10;   
const LOAD_MORE_BATCH   = 20;   
const width  = window.innerWidth;
const height = window.innerHeight;
const dx     = 38;
const dy     = 320;
let svg, g, zoom;
let i = 0;
const BOOT_LINES = [
  { text: '> INITIALIZING ICECAT TAXONOMY SYSTEM...', delay: 0,    cls: '' },
  { text: '> LOADING KERNEL MODULES [OK]',           delay: 300,  cls: 'ok' },
  { text: '> MOUNTING DATA FILESYSTEM...',           delay: 600,  cls: '' },
  { text: '> CONNECTING TO CLUSTER DATABASE [OK]',  delay: 900,  cls: 'ok' },
  { text: '> PARSING 20,000+ PRODUCT CLUSTERS...',  delay: 1200, cls: 'warn' },
  { text: '> BUILDING TAXONOMY TREE [OK]',          delay: 1600, cls: 'ok' },
  { text: '> RENDERING VISUALIZATION...',           delay: 2000, cls: '' },
];
function runBootSequence() {
  const container = document.getElementById('boot-lines');
  BOOT_LINES.forEach(({ text, delay, cls }) => {
    setTimeout(() => {
      const div = document.createElement('div');
      div.className = 'boot-line' + (cls ? ` ${cls}` : '');
      div.textContent = text;
      container.appendChild(div);
    }, delay);
  });
  setTimeout(() => {
    const wrap = document.getElementById('boot-launch-wrap');
    wrap.style.display = 'flex';
    document.getElementById('boot-cursor').style.display = 'none';
    const btn = document.getElementById('boot-launch-btn');
    btn.style.animation = 'launchBlink 1.2s ease-in-out infinite';
  }, 2600);
}
window.launchApp = function () {
  const overlay = document.getElementById('boot-overlay');
  overlay.classList.add('fade-out');
  setTimeout(() => {
    overlay.style.display = 'none';
    loadData();  
  }, 500);
};
document.addEventListener('DOMContentLoaded', () => {
  initViz();
  runBootSequence();
});
function initViz() {
  svg = d3.select('#tree-viz').append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('viewBox', [0, 0, width, height])
    .style('font', '12px "JetBrains Mono", monospace')
    .style('user-select', 'none');
  zoom = d3.zoom()
    .extent([[0, 0], [width, height]])
    .scaleExtent([0.05, 5])
    .on('zoom', (event) => {
      g.attr('transform', event.transform);
      const pct = Math.round(event.transform.k * 100);
      document.getElementById('zoom-indicator').textContent = `ZOOM: ${pct}%`;
    });
  svg.call(zoom);
  g = svg.append('g')
    .attr('transform', `translate(${dy / 2},${height / 2})`);
}
async function loadData() {
  try {
    const resp = await fetch('outputs/cluster_data.json');
    clusterData = await resp.json();
    try {
      const richResp = await fetch('outputs/cluster_data_rich.json');
      richData = await richResp.json();
    } catch (e) { console.warn('Rich data not found'); }
    processData();
    updateStats();
    renderTree(rootData);
    initCharts(clusterData, groupedData);
  } catch (e) {
    console.error(e);
    alert('SYSTEM ERROR: ' + e.message);
  }
}
function processData() {
  groupedData = {};
  clusterData.forEach(c => {
    if (!groupedData[c.dominant_category]) groupedData[c.dominant_category] = [];
    groupedData[c.dominant_category].push(c);
  });
  buildTreeData(currentPurity);
}
function buildTreeData(minPurity = 0) {
  const allCategories = Object.keys(groupedData).sort((a, b) => {
    const sizeA = groupedData[a].reduce((s, c) => s + c.size, 0);
    const sizeB = groupedData[b].reduce((s, c) => s + c.size, 0);
    return sizeB - sizeA;
  });
  const categories = categoryMode === 'top30'
    ? allCategories.slice(0, TOP_N)
    : allCategories;
  rootData = {
    name: 'Icecat Taxonomy',
    type: 'root',
    children: categories.map(catName => {
      const clusters = groupedData[catName].filter(c => c.purity >= minPurity);
      if (clusters.length === 0) return null;
      const totalSize = clusters.reduce((a, b) => a + b.size, 0);
      const clusterNodes = clusters.map(c => {
        const rich = richData[c.cluster_id] || richData[String(c.cluster_id)] || {};
        const breakdown = rich.breakdown || [];
        const topLabel = breakdown.length > 0
          ? breakdown[0].name
          : `Cluster #${c.cluster_id}`;
        return {
          name: topLabel,
          id: c.cluster_id,
          type: 'cluster',
          size: c.size,
          purity: c.purity,
          breakdown
        };
      });
      const visible  = clusterNodes.slice(0, INITIAL_CLUSTERS);
      const hidden   = clusterNodes.slice(INITIAL_CLUSTERS);
      const children = [...visible];
      if (hidden.length > 0) {
        children.push({
          name: `+ ${hidden.length} more clusters`,
          type: 'loadmore',
          catName,
          hidden   
        });
      }
      return {
        name: catName,
        type: 'category',
        size: totalSize,
        clusterCount: clusters.length,
        children
      };
    }).filter(Boolean)
  };
}
window.setCategoryMode = function (mode) {
  categoryMode = mode;
  document.getElementById('btn-top30').classList.toggle('active', mode === 'top30');
  document.getElementById('btn-all').classList.toggle('active', mode === 'all');
  i = 0;
  g.selectAll('*').remove();
  buildTreeData(currentPurity);
  renderTree(rootData);
  updateStats();
};
window.filterTree = function (percentage) {
  currentPurity = percentage / 100;
  i = 0;
  g.selectAll('*').remove();
  buildTreeData(currentPurity);
  renderTree(rootData);
  updateStatsCounts();
  zoomFit();
};
function updateStatsCounts() {
  const visibleClusters = rootData.children.reduce((s, cat) => s + (cat.children ? cat.children.length : 0), 0);
  document.getElementById('stat-categories').textContent = rootData.children.length.toLocaleString();
  document.getElementById('stat-clusters').textContent   = visibleClusters.toLocaleString();
}
function updateStats() {
  updateStatsCounts();
  const avgPurity = clusterData.reduce((s, c) => s + c.purity, 0) / clusterData.length;
  document.getElementById('stat-purity').textContent = (avgPurity * 100).toFixed(1) + '%';
}
function renderTree(data) {
  const root = d3.hierarchy(data);
  root.descendants().forEach(d => {
    if (d.depth > 0 && d.children) {
      d._children = d.children;
      d.children  = null;
    }
  });
  root.x0 = height / 2;
  root.y0 = 0;
  const tree = d3.tree().nodeSize([dx, dy]);
  update(root);
  function update(source) {
    const duration = 220;
    const nodes = root.descendants().reverse();
    const links  = root.links();
    tree(root);
    nodes.forEach(d => { d.y = d.depth * dy; });
    const node = g.selectAll('g.node')
      .data(nodes, d => d.id || (d.id = ++i));
    const nodeEnter = node.enter().append('g')
      .attr('class', 'node')
      .attr('transform', () => `translate(${source.y0},${source.x0})`)
      .on('click', (event, d) => click(event, d));
    nodeEnter.filter(d => d.data.type === 'root').each(function(d) {
      const el = d3.select(this);
      const RW = 240, RH = 52;
      el.append('rect')
        .attr('rx', 10).attr('ry', 10)
        .attr('width', RW + 12).attr('height', RH + 12)
        .attr('x', -6).attr('y', -(RH / 2) - 6)
        .style('fill', 'none')
        .style('stroke', '#00d4ff')
        .style('stroke-width', '1px')
        .style('filter', 'drop-shadow(0 0 12px #00d4ff)')
        .style('animation', 'rootPulse 2s ease-in-out infinite');
      el.append('rect')
        .attr('id', `node-${d.id}`)
        .attr('rx', 8).attr('ry', 8)
        .attr('width', RW).attr('height', RH)
        .attr('y', -RH / 2)
        .style('fill', 'rgba(0,212,255,0.15)')
        .style('stroke', '#00d4ff')
        .style('stroke-width', '1.5px')
        .style('cursor', 'pointer');
      const accent = (x, y, r) => el.append('circle').attr('cx', x).attr('cy', y).attr('r', r)
        .style('fill', '#00d4ff').style('filter', 'drop-shadow(0 0 4px #00d4ff)');
      accent(0, 0, 4); accent(RW, 0, 4);
      el.append('text').attr('x', 14).attr('dy', '0.35em')
        .style('fill', '#00d4ff').style('font-size', '16px')
        .style('font-family', 'FontAwesome').text('\uf0e8');
      el.append('text').attr('x', 38).attr('dy', '-0.15em')
        .style('fill', '#00d4ff').style('font-size', '13px').style('font-weight', '700')
        .style('font-family', '"JetBrains Mono", monospace').style('letter-spacing', '1px')
        .style('text-shadow', '0 0 10px #00d4ff')
        .text('Icecat Taxonomy');
      el.append('text').attr('x', 38).attr('dy', '1.1em')
        .style('fill', 'rgba(0,212,255,0.6)').style('font-size', '8px')
        .style('font-family', '"JetBrains Mono", monospace').style('letter-spacing', '3px')
        .text('CLUSTER EXPLORER v4.0');
      el.append('text').attr('x', RW - 16).attr('dy', '0.35em')
        .style('fill', '#00d4ff').style('font-size', '12px')
        .style('pointer-events', 'none')
        .text(d._children ? '▶' : '▼');
    });
    const nonRoot = nodeEnter.filter(d => d.data.type !== 'root');
    nonRoot.append('rect')
      .attr('class', 'node-glow')
      .attr('rx', 4).attr('ry', 4)
      .attr('width', 190).attr('height', 32)
      .attr('y', -16)
      .style('fill', 'none')
      .style('stroke', d => getNodeColorRaw(d))
      .style('stroke-width', '1px')
      .style('filter', d => `drop-shadow(0 0 5px ${getNodeColorRaw(d)})`);
    nonRoot.append('rect')
      .attr('rx', 4).attr('ry', 4)
      .attr('width', 190).attr('height', 32)
      .attr('y', -16)
      .attr('id', d => `node-${d.id}`)
      .style('fill', d => getNodeFill(d))
      .style('stroke', d => getNodeColorRaw(d))
      .style('stroke-width', '1px')
      .style('cursor', 'pointer');
    nonRoot.append('text')
      .attr('class', 'node-chevron')
      .attr('x', 174)
      .attr('dy', '0.35em')
      .style('fill', d => getNodeColorRaw(d))
      .style('font-size', '10px')
      .style('pointer-events', 'none')
      .text(d => (d._children ? '▶' : (d.children ? '▼' : '')));
    nonRoot.append('text')
      .attr('dy', '0.35em')
      .attr('x', 10)
      .style('text-anchor', 'start')
      .style('fill', d => getNodeColorRaw(d))
      .style('font-size', '10px')
      .style('font-family', '"JetBrains Mono", monospace')
      .style('pointer-events', 'none')
      .text(d => {
        const n = d.data.name;
        return n.length > 22 ? n.substring(0, 20) + '…' : n;
      });
    nodeEnter.filter(d => d.data.type === 'cluster').append('text')
      .attr('dy', '0.35em')
      .attr('x', 155)
      .style('text-anchor', 'end')
      .style('fill', d => getNodeColorRaw(d))
      .style('font-size', '9px')
      .style('font-family', '"JetBrains Mono", monospace')
      .style('pointer-events', 'none')
      .text(d => d.data.purity ? (d.data.purity * 100).toFixed(0) + '%' : '');
    const nodeUpdate = node.merge(nodeEnter).transition().duration(duration)
      .attr('transform', d => `translate(${d.y},${d.x})`);
    nodeUpdate.select('rect:not(.node-glow)')
      .style('fill', d => getNodeFill(d))
      .style('stroke', d => getNodeColorRaw(d));
    nodeUpdate.select('.node-glow')
      .style('stroke', d => getNodeColorRaw(d))
      .style('filter', d => `drop-shadow(0 0 5px ${getNodeColorRaw(d)})`);
    nodeUpdate.select('.node-chevron')
      .style('fill', d => getNodeColorRaw(d))
      .text(d => (d._children ? '▶' : (d.children ? '▼' : '')));
    node.exit().transition().duration(duration)
      .attr('transform', () => `translate(${source.y},${source.x})`)
      .remove();
    const link = g.selectAll('path.link')
      .data(links, d => d.target.id);
    const linkEnter = link.enter().insert('path', 'g')
      .attr('class', 'link')
      .attr('d', () => {
        const o = { x: source.x0, y: source.y0 };
        return diagonal(o, o);
      })
      .style('stroke', d => {
        const t = d.target.data.type;
        if (t === 'category') return 'rgba(0,255,65,0.3)';
        if (t === 'cluster')  return 'rgba(0,212,255,0.25)';
        return 'rgba(26,92,42,0.4)';
      });
    link.merge(linkEnter).transition().duration(duration)
      .attr('d', d => diagonal(d.source, d.target));
    link.exit().transition().duration(duration)
      .attr('d', () => {
        const o = { x: source.x, y: source.y };
        return diagonal(o, o);
      }).remove();
    nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
  }
  window.treeUpdate = update;
  window.treeRoot   = root;
}
const NODE_W      = 190; 
const ROOT_NODE_W = 240; 
function nodeWidth(d) {
  return (d && d.data && d.data.type === 'root') ? ROOT_NODE_W : NODE_W;
}
function diagonal(s, d) {
  const sx   = s.y + nodeWidth(s);
  const tx   = d.y;  
  const midX = (sx + tx) / 2;
  return `M ${sx} ${s.x}
          C ${midX} ${s.x},
            ${midX} ${d.x},
            ${tx} ${d.x}`;
}
function click(event, d) {
  if (d.data.type === 'loadmore') {
    const parent   = d.parent;
    const hidden   = d.data.hidden;
    const nextBatch = hidden.slice(0, LOAD_MORE_BATCH);
    const remaining = hidden.slice(LOAD_MORE_BATCH);
    const lmIdx = parent.children.findIndex(c => c.data.type === 'loadmore');
    if (lmIdx !== -1) {
      const rawChildren = parent.data.children;
      const rawLmIdx    = rawChildren.findIndex(c => c.type === 'loadmore');
      if (rawLmIdx !== -1) {
        rawChildren.splice(rawLmIdx, 1, ...nextBatch);
        if (remaining.length > 0) {
          rawChildren.push({ name: `+ ${remaining.length} more clusters`, type: 'loadmore', catName: d.data.catName, hidden: remaining });
        }
      }
    }
    i = 0;
    g.selectAll('*').remove();
    renderTree(rootData);
    return;
  }
  showDetails(d.data);
  if (d.children) {
    d._children = d.children;
    d.children  = null;
  } else {
    d.children  = d._children;
    d._children = null;
  }
  window.treeUpdate(d);
}
window.collapseAll = function () {
  const root = window.treeRoot;
  if (!root) return;
  root.descendants().forEach(d => {
    if (d.depth > 0 && d.children) {
      d._children = d.children;
      d.children  = null;
    }
  });
  window.treeUpdate(root);
  zoomFit();
};
window.expandCategories = function () {
  const root = window.treeRoot;
  if (!root) return;
  root.descendants().forEach(d => {
    if (d.depth === 0 && d._children) { d.children = d._children; d._children = null; }
    if (d.depth === 1 && d._children) { d.children = d._children; d._children = null; }
    if (d.depth >= 2 && d.children)   { d._children = d.children; d.children  = null; }
  });
  window.treeUpdate(root);
  zoomFit();
};
function getNodeColorRaw(d) {
  const t = d.data.type;
  if (t === 'root')     return '#00d4ff';
  if (t === 'category') return '#00ff41';
  if (t === 'cluster') {
    const p = d.data.purity || 0;
    return p >= 0.9 ? '#00ff41' : (p >= 0.7 ? '#f5a623' : '#ff2a6d');
  }
  if (t === 'loadmore') return '#00d4ff'; 
  return '#1a5c2a';
}
function getNodeFill(d) {
  const t = d.data.type;
  if (t === 'root')     return 'rgba(0,212,255,0.12)';
  if (t === 'category') return d._children ? 'rgba(0,255,65,0.10)' : 'rgba(0,255,65,0.06)';
  if (t === 'cluster') {
    const p = d.data.purity || 0;
    if (p >= 0.9) return 'rgba(0,255,65,0.10)';
    if (p >= 0.7) return 'rgba(245,166,35,0.10)';
    return 'rgba(255,42,109,0.10)';
  }
  if (t === 'loadmore') return 'rgba(0,212,255,0.06)';
  return 'rgba(10,30,15,0.7)';
}
let activeDonutChart = null; 
function getDonutColor(index) {
  const palette = [
    '#00ff41', '#00d4ff', '#f5a623', '#ff2a6d',
    '#a78bfa', '#34d399', '#fb923c', '#60a5fa',
    '#f472b6', '#4ade80',
  ];
  return palette[index % palette.length];
}
function showDetails(data) {
  detailsPanel.classList.add('open');
  if (activeDonutChart) { activeDonutChart.destroy(); activeDonutChart = null; }
  if (data.type === 'cluster') {
    const purity    = (data.purity * 100).toFixed(1);
    const cls       = data.purity >= 0.9 ? 'hi' : (data.purity >= 0.7 ? 'mid' : 'lo');
    const clsLabel  = data.purity >= 0.9 ? 'HIGH PURITY' : (data.purity >= 0.7 ? 'MED PURITY' : 'LOW PURITY');
    const breakdown = data.breakdown || [];
    detailsContent.innerHTML = `
      <div class="d-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
          <span class="d-card-title">#${data.id}</span>
          <span class="d-purity-badge ${cls}">${purity}%</span>
        </div>
        <div class="d-meta">
          <i class="fas fa-layer-group" style="color:var(--green-dim)"></i>
          <span>${data.size.toLocaleString()} products</span>
          &nbsp;·&nbsp;
          <span style="color:var(--text-dim); font-size:9px;">${clsLabel}</span>
        </div>
      </div>
      <div class="d-section-label">
      ${breakdown.length > 0 ? `
        <div style="position:relative; width:100%; padding-bottom:4px;">
          <canvas id="clusterDonutChart" height="200"></canvas>
        </div>
        <div style="margin-top:10px; display:flex; flex-direction:column; gap:6px;">
          ${breakdown.map((b, i) => `
            <div style="display:flex; align-items:center; gap:8px; font-size:10px;">
              <span style="width:10px; height:10px; border-radius:2px; flex-shrink:0; background:${getDonutColor(i)};"></span>
              <span style="flex:1; color:var(--text-primary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${b.name}">${b.name}</span>
              <span style="color:var(--text-dim); font-size:9px; flex-shrink:0;">${b.count.toLocaleString()}p</span>
              <span style="color:var(--text-secondary); font-size:10px; font-weight:600; flex-shrink:0; min-width:34px; text-align:right;">${b.percentage.toFixed(1)}%</span>
            </div>`).join('')}
        </div>
      ` : `
        <div style="text-align:center; padding:24px; color:var(--text-dim); font-size:10px; border:1px dashed var(--border); border-radius:4px; letter-spacing:2px;">
        </div>`}
    `;
    if (breakdown.length > 0) {
      const ctx = document.getElementById('clusterDonutChart').getContext('2d');
      activeDonutChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: breakdown.map(b => b.name),
          datasets: [{
            data: breakdown.map(b => b.percentage),
            backgroundColor: breakdown.map((_, i) => getDonutColor(i) + 'cc'), 
            borderColor: breakdown.map((_, i) => getDonutColor(i)),
            borderWidth: 1.5,
            hoverOffset: 8
          }]
        },
        options: {
          responsive: true,
          cutout: '60%',
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#030f07',
              borderColor: '#0d3d1a',
              borderWidth: 1,
              titleColor: '#00ff41',
              bodyColor: '#5a9e6f',
              titleFont: { family: "'JetBrains Mono', monospace", size: 10 },
              bodyFont:  { family: "'JetBrains Mono', monospace", size: 10 },
              callbacks: {
                label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`
              }
            }
          },
          animation: { animateRotate: true, duration: 450 }
        },
        plugins: [{
          id: 'centerText',
          afterDraw(chart) {
            const { ctx: c, chartArea: { left, right, top, bottom } } = chart;
            const cx = (left + right) / 2;
            const cy = (top + bottom) / 2;
            const color = data.purity >= 0.9 ? '#00ff41' : (data.purity >= 0.7 ? '#f5a623' : '#ff2a6d');
            c.save();
            c.textAlign = 'center'; c.textBaseline = 'middle';
            c.font = `bold 20px 'JetBrains Mono', monospace`;
            c.fillStyle = color;
            c.shadowColor = color; c.shadowBlur = 12;
            c.fillText(purity + '%', cx, cy - 9);
            c.font = `9px 'JetBrains Mono', monospace`;
            c.fillStyle = '#5a9e6f'; c.shadowBlur = 0;
            c.fillText('PURITY', cx, cy + 11);
            c.restore();
          }
        }]
      });
    }
  } else if (data.type === 'category') {
    detailsContent.innerHTML = `
      <div class="d-card" style="border-color: var(--green-dim); background: rgba(0,255,65,0.05);">
        <div class="d-card-title" style="font-size:16px; margin-bottom:8px;">${data.name}</div>
        <div class="d-meta">
          <i class="fas fa-folder-open" style="color:var(--green-dim)"></i>
          <span>${data.clusterCount || (data.children ? data.children.length : 0)} clusters</span>
          &nbsp;·&nbsp;
          <span>${data.size ? data.size.toLocaleString() : '—'} products</span>
        </div>
      </div>
      <div class="d-section-label">
      <div style="font-size:10px; color:var(--text-secondary); line-height:1.9; padding: 10px 0;">
        &gt; Dominant category group derived from<br>
        &nbsp;&nbsp;&nbsp;cluster label assignment.<br><br>
        &gt; Click child nodes (green) to explore<br>
        &nbsp;&nbsp;&nbsp;individual product clusters.<br><br>
        &gt; Purity filter above can be used to<br>
        &nbsp;&nbsp;&nbsp;show only high-confidence clusters.
      </div>
    `;
  } else {
    detailsContent.innerHTML = `
      <div class="details-empty">
        <i class="fas fa-crosshairs"></i>
        <p>Select a node to inspect</p>
      </div>`;
  }
}
function toggleDetails() {
  detailsPanel.classList.toggle('open');
}
function zoomIn()  { svg.transition().call(zoom.scaleBy, 1.25); }
function zoomOut() { svg.transition().call(zoom.scaleBy, 0.8);  }
function zoomFit() {
  svg.transition().duration(700).call(zoom.transform, d3.zoomIdentity.translate(dy / 2, height / 2));
}
const searchInput    = document.getElementById('searchInput');
const suggestionsBox = document.getElementById('searchSuggestions');
searchInput.addEventListener('input', (e) => {
  const term = e.target.value.toLowerCase().trim();
  suggestionsBox.innerHTML = '';
  if (term.length < 2) {
    suggestionsBox.classList.remove('visible');
    return;
  }
  const results = [];
  rootData.children.forEach(cat => {
    if (cat.name.toLowerCase().includes(term))
      results.push({ type: 'CATEGORY', name: cat.name, data: cat });
    if (cat.children) {
      cat.children
        .filter(c => c.name.toLowerCase().includes(term) || String(c.id).includes(term))
        .slice(0, 5)
        .forEach(c => results.push({ type: 'CLUSTER', name: c.name, data: c }));
    }
  });
  if (results.length > 0) {
    suggestionsBox.classList.add('visible');
    results.slice(0, 20).forEach(res => {
      const li = document.createElement('li');
      li.innerHTML = `<span>${res.name}</span><span class="type-badge">${res.type}</span>`;
      li.onclick = () => {
        selectNode(res.data);
        suggestionsBox.classList.remove('visible');
        searchInput.value = res.name;
      };
      suggestionsBox.appendChild(li);
    });
  } else {
    suggestionsBox.classList.remove('visible');
  }
});
document.addEventListener('click', (e) => {
  if (!searchInput.contains(e.target) && !suggestionsBox.contains(e.target))
    suggestionsBox.classList.remove('visible');
});
function searchTree() {
  const term = searchInput.value.toLowerCase();
  if (!term) return;
  const root  = window.treeRoot;
  const match = root.descendants().find(d => d.data.name.toLowerCase().includes(term));
  if (match) selectNode(match.data);
}
function selectNode(targetData) {
  const root = window.treeRoot;
  let match  = null;
  if (targetData.type === 'cluster') {
    match = root.descendants().find(d => d.data.id === targetData.id && d.data.type === 'cluster');
  } else {
    match = root.descendants().find(d => d.data.name === targetData.name && d.data.type === 'category');
  }
  if (!match) return;
  let p = match;
  while (p.parent) {
    if (p.parent._children) {
      p.parent.children  = p.parent._children;
      p.parent._children = null;
    }
    p = p.parent;
  }
  window.treeUpdate(match);
  const t = d3.zoomIdentity
    .translate(width / 2 - match.y, height / 2 - match.x)
    .scale(1.6);
  svg.transition().duration(900).call(zoom.transform, t);
  d3.selectAll('.node rect:not(.node-glow)').style('stroke', null).style('stroke-width', '1px');
  d3.select(`#node-${match.id}`)
    .style('stroke', '#ffffff')
    .style('stroke-width', '2.5px');
  showDetails(match.data);
}
window.exportTree = function () {
  const svgEl  = document.querySelector('#tree-viz svg');
  const serial = new XMLSerializer();
  const str    = serial.serializeToString(svgEl);
  const canvas = document.createElement('canvas');
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#020c06';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const img  = new Image();
  const blob = new Blob([str], { type: 'image/svg+xml;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  img.onload = () => {
    ctx.drawImage(img, 0, 0);
    const a = document.createElement('a');
    a.download = 'icecat-taxonomy-hacker.png';
    a.href = canvas.toDataURL('image/png');
    a.click();
    URL.revokeObjectURL(url);
  };
  img.src = url;
};