
// Logic for Icecat Taxonomy Explorer v4.0

const vizContainer = document.getElementById('viz-container');
const detailsPanel = document.getElementById('details-panel');
const detailsContent = document.getElementById('details-content');

let rootData = null;
let clusterData = [];
let richData = {};
let groupedData = {};

// D3 Configuration
const width = window.innerWidth;
const height = window.innerHeight;
const dx = 40; // Vertical spacing
const dy = 200; // Horizontal spacing

let svg, g, zoom;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initViz();
    loadData();
});

function initViz() {
    svg = d3.select("#tree-viz").append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("viewBox", [0, 0, width, height])
        .style("font", "12px sans-serif")
        .style("user-select", "none");

    // Zoom behavior
    zoom = d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        });

    svg.call(zoom);

    g = svg.append("g")
        .attr("transform", `translate(${dy / 2},${height / 2})`);
}

async function loadData() {
    try {
        console.log("Loading data...");
        // 1. Load Cluster Data
        const resp = await fetch('outputs/cluster_data.json');
        clusterData = await resp.json();

        // 2. Load Rich Data (Optional)
        try {
            const richResp = await fetch('outputs/cluster_data_rich.json');
            richData = await richResp.json();
        } catch (e) { console.warn("Rich data not found"); }

        // 3. Process
        processData();

        // 4. Update Stats in Header
        updateStats();

        // 5. Render
        renderTree(rootData);
        initCharts(clusterData);
    } catch (e) {
        console.error(e);
        alert("Failed to load data: " + e.message);
    }
}

function processData() {
    // Group by Dominant Category
    groupedData = {};
    clusterData.forEach(c => {
        if (!groupedData[c.dominant_category]) groupedData[c.dominant_category] = [];
        groupedData[c.dominant_category].push(c);
    });

    // Build initial tree
    buildTreeData(0);
}

function buildTreeData(minPurity = 0) {
    const categories = Object.keys(groupedData).sort();

    rootData = {
        name: "Icecat Taxonomy",
        type: "root",
        children: categories.map(catName => {
            // Filter clusters by purity
            const clusters = groupedData[catName].filter(c => c.purity >= minPurity);

            if (clusters.length === 0) return null;

            return {
                name: catName,
                type: "category",
                size: clusters.reduce((a, b) => a + b.size, 0),
                children: clusters.map(c => ({
                    name: `Cluster ${c.cluster_id}`,
                    id: c.cluster_id,
                    type: "cluster",
                    size: c.size,
                    purity: c.purity,
                    // Sub-children (Breakdown)
                    children: (richData[c.cluster_id] || richData[String(c.cluster_id)] || {}).breakdown?.map(b => ({
                        name: `${b.name}`,
                        type: "leaf",
                        percentage: b.percentage,
                        count: b.count
                    }))
                }))
            };
        }).filter(cat => cat !== null) // Remove empty categories
    };
}

// Global Filter Function
window.filterTree = function (percentage) {
    const threshold = percentage / 100;
    buildTreeData(threshold);
    renderTree(rootData);

    // Update stats based on filtered view
    // We need to count visible clusters
    const visibleClusters = rootData.children.reduce((sum, cat) => sum + (cat.children ? cat.children.length : 0), 0);
    document.getElementById('stat-categories').innerText = rootData.children.length.toLocaleString();
    document.getElementById('stat-clusters').innerText = visibleClusters.toLocaleString();
};

function renderTree(data) {
    const root = d3.hierarchy(data);

    // Initial Collapse: Collapse everything at depth > 0 (Categories collapsed)
    root.descendants().forEach((d) => {
        if (d.depth > 0 && d.children) {
            d._children = d.children;
            d.children = null;
        }
    });

    // Initial Position
    root.x0 = height / 2;
    root.y0 = 0;

    const tree = d3.tree().nodeSize([dx, dy]);

    update(root);

    function update(source) {
        const duration = 250;
        const nodes = root.descendants().reverse();
        const links = root.links();

        // Compute Layout
        tree(root);

        // Normalize for fixed depth.
        nodes.forEach(d => { d.y = d.depth * dy; });

        // Node
        const node = g.selectAll('g.node')
            .data(nodes, d => d.id || (d.id = ++i)); // Generate IDs if missing

        const nodeEnter = node.enter().append('g')
            .attr('class', 'node')
            .attr("transform", d => `translate(${source.y0},${source.x0})`)
            .on("click", (event, d) => click(event, d));

        // Rects
        nodeEnter.append('rect')
            .attr('rx', 6)
            .attr('ry', 6)
            .attr('width', 180)
            .attr('height', 30)
            .attr('y', -15)
            .attr('id', d => `node-${d.id}`) // for search highlight
            .style("fill", d => getNodeColor(d))
            .style("stroke", "#fff")
            .style("stroke-width", "1px")
            .style("cursor", "pointer");

        // Text
        nodeEnter.append('text')
            .attr("dy", ".35em")
            .attr("x", 10)
            .style("text-anchor", "start")
            .text(d => d.data.name.length > 25 ? d.data.name.substring(0, 22) + "..." : d.data.name)
            .style("fill", "white")
            .style("font-size", "11px")
            .style("pointer-events", "none"); // Let clicks pass to rect

        // Transition Nodes
        const nodeUpdate = node.merge(nodeEnter).transition().duration(duration)
            .attr("transform", d => `translate(${d.y},${d.x})`);

        nodeUpdate.select('rect')
            .style("fill", d => getNodeColor(d));

        const nodeExit = node.exit().transition().duration(duration)
            .attr("transform", d => `translate(${source.y},${source.x})`)
            .remove();

        // Links
        const link = g.selectAll('path.link')
            .data(links, d => d.target.id);

        const linkEnter = link.enter().insert('path', "g")
            .attr("class", "link")
            .attr('d', d => {
                const o = { x: source.x0, y: source.y0 };
                return diagonal(o, o);
            });

        link.merge(linkEnter).transition().duration(duration)
            .attr('d', d => diagonal(d.source, d.target));

        link.exit().transition().duration(duration)
            .attr('d', d => {
                const o = { x: source.x, y: source.y };
                return diagonal(o, o);
            })
            .remove();

        // Stash positions
        nodes.forEach(d => {
            d.x0 = d.x;
            d.y0 = d.y;
        });
    }

    // Store update function globally for zooming/searching? 
    // No, cleaner to keep it scoped. But search might need access. 
    // For now, simple tree.
    window.treeUpdate = update;
    window.treeRoot = root;
}

let i = 0;

function diagonal(s, d) {
    return `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
              ${(s.y + d.y) / 2} ${d.x},
              ${d.y} ${d.x}`;
}

function click(event, d) {
    // 1. Show Details in Panel
    showDetails(d.data);

    // 2. Toggle Children
    if (d.children) {
        d._children = d.children;
        d.children = null;
    } else {
        d.children = d._children;
        d._children = null;
    }
    window.treeUpdate(d);
}

function getNodeColor(d) {
    // Purity based color?
    if (d.data.type === 'root') return '#2563eb'; // Blue
    if (d.data.type === 'category') return d._children ? '#059669' : '#10b981'; // Green Dark/Light
    if (d.data.type === 'cluster') {
        const p = d.data.purity || 0;
        return p > 0.9 ? '#10b981' : (p > 0.8 ? '#f59e0b' : '#ef4444');
    }
    return '#6b7280'; // Grey Leaf
}

// Stats & UI
function updateStats() {
    document.getElementById('stat-categories').innerText = Object.keys(groupedData).length.toLocaleString();
    document.getElementById('stat-clusters').innerText = clusterData.length.toLocaleString();
    const avgPurity = clusterData.reduce((sum, c) => sum + c.purity, 0) / clusterData.length;
    document.getElementById('stat-purity').innerText = (avgPurity * 100).toFixed(1) + '%';
}

function showDetails(data) {
    if (detailsPanel.classList.contains('translate-x-full')) {
        detailsPanel.classList.remove('translate-x-full');
    }

    if (data.type === 'cluster') {
        const purity = (data.purity * 100).toFixed(1);
        const color = data.purity > 0.9 ? 'text-green-400' : (data.purity > 0.8 ? 'text-yellow-400' : 'text-red-400');

        let html = `
            <div class="bg-gray-700 p-5 rounded-xl mb-6 shadow-lg border border-gray-600">
                <div class="flex justify-between items-start mb-2">
                    <h4 class="text-2xl font-bold text-white leading-tight">Cluster #${data.id}</h4>
                    <span class="${color} font-mono font-bold bg-gray-800 px-2 py-1 rounded text-sm">${purity}%</span>
                </div>
                <div class="text-sm text-gray-400 flex items-center">
                    <i class="fas fa-box-open mr-2"></i> ${data.size.toLocaleString()} products
                </div>
            </div>
            
            <h5 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 border-b border-gray-700 pb-2">
                Semantic Composition (Word Cloud)
            </h5>
        `;

        if (data.children && data.children.length > 0) {
            // Word Cloud Container
            html += `<div class="flex flex-wrap gap-2 content-start">`;

            // Sort by percentage descending for cloud logic
            const items = [...data.children].sort((a, b) => b.percentage - a.percentage);

            items.forEach(item => {
                // Calculate size/color
                // Base size 0.75rem. Max size 1.5rem.
                // Percentage 0-100.
                // Log scale might be better? Linear for now.
                const scale = Math.min(item.percentage / 20, 1.5) + 0.75;
                const fontSize = `${scale.toFixed(2)}rem`;

                // Opacity based on significance
                const opacity = Math.max(item.percentage / 40, 0.4);
                const textColor = item.percentage > 40 ? 'text-blue-300' : 'text-gray-300';
                const bgClass = item.percentage > 40 ? 'bg-blue-900/50 border-blue-700' : 'bg-gray-700/50 border-gray-600';

                html += `
                    <div class="${bgClass} border rounded-lg px-3 py-1 flex items-center transition hover:scale-105 cursor-default" title="${item.count} items (${item.percentage.toFixed(1)}%)">
                        <span class="${textColor} font-medium" style="font-size: 0.85rem;">${item.name}</span>
                        <span class="ml-2 text-xs text-gray-400 bg-gray-800 px-1 rounded">${item.percentage.toFixed(0)}%</span>
                    </div>
                 `;
            });
            html += `</div>`;

            // Add "Product Samples" placeholder if we had them
            // html += `<div class="mt-6 text-xs text-gray-500 italic"><i class="fas fa-info-circle mr-1"></i> Based on top subcategories</div>`

        } else {
            html += `<div class="text-center py-8 text-gray-500 italic border-2 border-dashed border-gray-700 rounded-lg">
                <i class="fas fa-ghost text-2xl mb-2 opacity-50"></i><br>
                No breakdown details available.
            </div>`;
        }

        detailsContent.innerHTML = html;

    } else if (data.type === 'category') {
        detailsContent.innerHTML = `
            <div class="bg-blue-900 border border-blue-700 p-6 rounded-xl shadow-lg relative overflow-hidden">
                <div class="absolute -right-4 -top-4 text-9xl text-blue-800 opacity-20 transform rotate-12">
                    <i class="fas fa-folder"></i>
                </div>
                <h4 class="text-3xl font-bold text-white mb-2 relative z-10">${data.name}</h4>
                <div class="flex items-center space-x-4 mt-4 relative z-10">
                    <div class="bg-blue-800 px-3 py-1 rounded-lg text-sm text-blue-100 font-medium">
                        <i class="fas fa-layer-group mr-1"></i> Group
                    </div>
                    <div class="text-blue-200 text-sm">
                        ${data.children ? data.children.length : 0} Clusters
                    </div>
                </div>
            </div>
            
            <div class="mt-6 p-4 bg-gray-800 rounded-lg border border-gray-700">
                <p class="text-gray-400 text-sm leading-relaxed">
                    This is a high-level category group derived from the dominant labels in the clustered data.
                    Click on the green nodes attached to this group to explore specific product clusters.
                </p>
            </div>
        `;
    } else {
        detailsContent.innerHTML = `<div class="flex flex-col items-center justify-center h-64 text-gray-500">
            <i class="fas fa-mouse-pointer text-4xl mb-4 opacity-30 animate-bounce"></i>
            <p>Select a node to inspect</p>
        </div>`;
    }
}

function toggleDetails() {
    detailsPanel.classList.toggle('translate-x-full');
}

function zoomIn() {
    svg.transition().call(zoom.scaleBy, 1.2);
}

function zoomOut() {
    svg.transition().call(zoom.scaleBy, 0.8);
}

function zoomFit() {
    // Reset zoom
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
}

// Search Logic
const searchInput = document.getElementById('searchInput');
const suggestionsBox = document.getElementById('searchSuggestions');

searchInput.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    suggestionsBox.innerHTML = '';

    if (term.length < 2) {
        suggestionsBox.classList.add('hidden');
        return;
    }

    // Search in Categories and Clusters
    // We can search rootData structure or clusterData array
    // Searching rootData is better to find Categories
    const results = [];

    // 1. Search Categories
    rootData.children.forEach(cat => {
        if (cat.name.toLowerCase().includes(term)) {
            results.push({ type: 'Category', name: cat.name, data: cat });
        }
        // 2. Search Clusters (limited to top 10 per category to avoid spam)
        if (cat.children) {
            const matchingClusters = cat.children.filter(c => c.name.toLowerCase().includes(term) || String(c.id).includes(term));
            matchingClusters.slice(0, 5).forEach(c => {
                results.push({ type: 'Cluster', name: c.name, data: c });
            });
        }
    });

    if (results.length > 0) {
        suggestionsBox.classList.remove('hidden');
        // Limit to 20 results
        results.slice(0, 20).forEach(res => {
            const li = document.createElement('li');
            li.className = "px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between";
            li.innerHTML = `<span>${res.name}</span> <span class="text-xs text-gray-500 uppercase">${res.type}</span>`;
            li.onclick = () => {
                selectNode(res.data);
                suggestionsBox.classList.add('hidden');
                searchInput.value = res.name;
            };
            suggestionsBox.appendChild(li);
        });
    } else {
        suggestionsBox.classList.add('hidden');
    }
});

// Hide suggestions on click outside
document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !suggestionsBox.contains(e.target)) {
        suggestionsBox.classList.add('hidden');
    }
});

function searchTree() {
    // Legacy button handler - just trigger first result if any?
    const term = searchInput.value.toLowerCase();
    if (!term) return;

    // Same logic as input, select first
    const root = window.treeRoot; // Use D3 hierarchy to find node
    const match = root.descendants().find(d => d.data.name.toLowerCase().includes(term));
    if (match) selectNode(match.data);
}

function selectNode(targetData) {
    const root = window.treeRoot;
    // Find d3 node by ID or Name
    let match = null;

    // If it's a cluster, we have ID
    if (targetData.type === 'cluster') {
        match = root.descendants().find(d => d.data.id === targetData.id && d.data.type === 'cluster');
    } else {
        match = root.descendants().find(d => d.data.name === targetData.name && d.data.type === 'category');
    }

    if (match) {
        // Expand path to match
        let p = match;
        while (p.parent) {
            if (p.parent.data.children && !p.parent.children) {
                // It's collapsed. Expand it.
                // We need to restore children from stashed _children if available,
                // or rebuild if we filter. 
                // In our simple toggle logic:
                if (p.parent._children) {
                    p.parent.children = p.parent._children;
                    p.parent._children = null;
                }
            }
            p = p.parent;
        }
        window.treeUpdate(match);

        // Zoom and Highlight
        const t = d3.zoomIdentity.translate(width / 2 - match.y, height / 2 - match.x).scale(1.5);
        svg.transition().duration(1000).call(zoom.transform, t);

        d3.selectAll('.node rect').style('stroke', '#fff').style('stroke-width', '1px');
        d3.select(`#node-${match.id}`).style('stroke', '#facc15').style('stroke-width', '4px'); // Yellow

        showDetails(match.data);
    }
}

// Export Logic
window.exportTree = function () {
    // Select the SVG
    const svgElement = document.querySelector("#tree-viz svg");

    // Serialize
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgElement);

    // Canvas
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const ctx = canvas.getContext('2d');

    // Background
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Image
    const img = new Image();
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);

    img.onload = () => {
        ctx.drawImage(img, 0, 0);

        const a = document.createElement('a');
        a.download = 'icecat-taxonomy-tree.png';
        a.href = canvas.toDataURL('image/png');
        a.click();

        URL.revokeObjectURL(url);
    };
    img.src = url;
};
