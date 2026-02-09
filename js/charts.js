
function initCharts(data) {
    const ctx = document.getElementById('miniPurityChart').getContext('2d');

    // Bucket purity into 10% bins
    const bins = new Array(10).fill(0);
    let total = 0;

    data.forEach(d => {
        const idx = Math.min(Math.floor(d.purity * 10), 9);
        bins[idx]++;
        total++;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'],
            datasets: [{
                label: 'Clusters',
                data: bins,
                backgroundColor: bins.map((_, i) => i > 7 ? '#10b981' : '#6b7280'),
                borderRadius: 4
            }]
        },
        options: {
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            maintainAspectRatio: false
        }
    });

    // Update Label logic
    document.getElementById('purityFilter').addEventListener('input', (e) => {
        const val = e.target.value;
        document.getElementById('purityLabel').innerText = val + '%';

        // Debounce or just call? Tree render is fast enough for <500 nodes, but maybe laggy for 20k.
        // Let's rely on D3 efficiency.
        if (window.filterTree) {
            window.filterTree(val);
        }
    });
}
