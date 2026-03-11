// ============================================================
// ICECAT TAXONOMY EXPLORER — HACKER CHARTS
// Chart.js: Purity Distribution + Top Categories
// ============================================================

function initCharts(data, groupedData) {
  // ── Chart 1: Purity Distribution Histogram ────────────────
  const bins = new Array(10).fill(0);
  data.forEach(d => {
    const idx = Math.min(Math.floor(d.purity * 10), 9);
    bins[idx]++;
  });

  const purityCtx = document.getElementById('miniPurityChart').getContext('2d');
  new Chart(purityCtx, {
    type: 'bar',
    data: {
      labels: ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90'],
      datasets: [{
        data: bins,
        backgroundColor: bins.map((_, i) =>
          i >= 9 ? 'rgba(0,255,65,0.85)' :
          i >= 7 ? 'rgba(0,255,65,0.55)' :
          i >= 5 ? 'rgba(245,166,35,0.5)' :
                   'rgba(255,42,109,0.45)'
        ),
        borderColor: bins.map((_, i) =>
          i >= 9 ? '#00ff41' :
          i >= 7 ? '#00b32d' :
          i >= 5 ? '#f5a623' :
                   '#ff2a6d'
        ),
        borderWidth: 1,
        borderRadius: 3
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#030f07',
          borderColor: '#0d3d1a',
          borderWidth: 1,
          titleColor: '#00ff41',
          bodyColor: '#5a9e6f',
          titleFont: { family: "'JetBrains Mono', monospace", size: 10 },
          bodyFont: { family: "'JetBrains Mono', monospace", size: 10 },
          callbacks: {
            title: (ctx) => `PURITY: ${ctx[0].label}–${parseInt(ctx[0].label)+10}%`,
            label: (ctx) => ` ${ctx.parsed.y.toLocaleString()} clusters`
          }
        }
      },
      scales: {
        x: {
          display: true,
          ticks: {
            color: '#2a5c3a',
            font: { family: "'JetBrains Mono', monospace", size: 8 },
            maxRotation: 0
          },
          grid: { color: 'rgba(13,61,26,0.4)' }
        },
        y: {
          display: true,
          ticks: {
            color: '#2a5c3a',
            font: { family: "'JetBrains Mono', monospace", size: 8 },
            maxTicksLimit: 4
          },
          grid: { color: 'rgba(13,61,26,0.4)' }
        }
      }
    }
  });

  // ── Chart 2: Top 5 Categories by Product Count ────────────
  const catTotals = Object.entries(groupedData)
    .map(([name, clusters]) => ({
      name: name.length > 18 ? name.slice(0, 16) + '…' : name,
      total: clusters.reduce((s, c) => s + c.size, 0)
    }))
    .sort((a, b) => b.total - a.total)
    .slice(0, 5);

  const topCatCtx = document.getElementById('miniTopCatChart').getContext('2d');
  new Chart(topCatCtx, {
    type: 'bar',
    data: {
      labels: catTotals.map(c => c.name),
      datasets: [{
        data: catTotals.map(c => c.total),
        backgroundColor: 'rgba(0,212,255,0.18)',
        borderColor: '#00d4ff',
        borderWidth: 1,
        borderRadius: 3
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#030f07',
          borderColor: '#0d3d1a',
          borderWidth: 1,
          titleColor: '#00d4ff',
          bodyColor: '#5a9e6f',
          titleFont: { family: "'JetBrains Mono', monospace", size: 10 },
          bodyFont: { family: "'JetBrains Mono', monospace", size: 10 },
          callbacks: {
            label: (ctx) => ` ${ctx.parsed.x.toLocaleString()} products`
          }
        }
      },
      scales: {
        x: {
          display: true,
          ticks: {
            color: '#2a5c3a',
            font: { family: "'JetBrains Mono', monospace", size: 8 },
            maxTicksLimit: 4,
            callback: val => val >= 1000 ? (val/1000).toFixed(0)+'k' : val
          },
          grid: { color: 'rgba(13,61,26,0.4)' }
        },
        y: {
          display: true,
          ticks: {
            color: '#00d4ff',
            font: { family: "'JetBrains Mono', monospace", size: 8 }
          },
          grid: { display: false }
        }
      }
    }
  });

  // ── Purity Slider ─────────────────────────────────────────
  const slider = document.getElementById('purityFilter');
  const label  = document.getElementById('purityLabel');

  slider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    label.textContent = val + '%';
    if (window.filterTree) window.filterTree(val);
  });
}
