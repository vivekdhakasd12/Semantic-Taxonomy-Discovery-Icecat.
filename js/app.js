
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

    // Build Hierarchy
    // Limit to top 50 categories for performance initially, or lazy load?
    // Let's take all categories but collapse them.
    const categories = Object.keys(groupedData).sort();

    rootData = {
        name: "Icecat Taxonomy",
        type: "root",
        children: categories.map(catName => {
            const clusters = groupedData[catName];
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
        })
    };
}

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
    // Open Panel if closed
    if (detailsPanel.classList.contains('translate-x-full')) {
        detailsPanel.classList.remove('translate-x-full');
    }

    if (data.type === 'cluster') {
        const breakdown = data.children; // Assuming simple array of breakdown
        // Wait, children in 'data' are the breakdown items if loaded
        // Or if data is from D3 node, data.children might be the node children logic.
        // We stored breakdown in data.children in processData()

        let html = `
            <div class="bg-gray-700 p-4 rounded-lg mb-4">
                <h4 class="text-xl font-bold text-white mb-1">Cluster #${data.id}</h4>
                <div class="text-sm text-gray-400">Purity: ${(data.purity * 100).toFixed(1)}%</div>
                <div class="text-sm text-gray-400">Size: ${data.size}</div>
            </div>
            
            <h5 class="text-md font-semibold text-gray-300 mb-3 border-b border-gray-600 pb-2">Top Categories in Cluster</h5>
        `;

        // If breakdown exists (it was put into children)
        if (data.children && data.children.length > 0) {
            html += `<div class="space-y-3">`;
            data.children.forEach(item => {
                const pct = (item.percentage).toFixed(1);
                html += `
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span class="text-gray-200">${item.name}</span>
                            <span class="text-gray-400">${item.count} (${pct}%)</span>
                        </div>
                        <div class="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div class="h-full bg-blue-500" style="width: ${Math.min(item.percentage, 100)}%"></div>
                        </div>
                    </div>
                 `;
            });
            html += `</div>`;
        } else {
            html += `<p class="text-gray-500 italic">No breakdown details available.</p>`;
        }

        detailsContent.innerHTML = html;

    } else if (data.type === 'category') {
        detailsContent.innerHTML = `
            <div class="bg-blue-900 p-4 rounded-lg">
                <h4 class="text-xl font-bold text-white">${data.name}</h4>
                <p class="text-sm text-blue-200">Category Group</p>
                <div class="mt-4 text-white">
                    Contains ${data.children ? data.children.length : 0} Clusters.
                </div>
            </div>
        `;
    } else {
        detailsContent.innerHTML = `<p class="text-gray-400">Select a Category or Cluster to view details.</p>`;
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
function searchTree() {
    const term = document.getElementById('searchInput').value.toLowerCase();
    if (!term) return;

    // Find node
    const root = window.treeRoot;
    const match = root.descendants().find(d => d.data.name.toLowerCase().includes(term));

    if (match) {
        // Expand path to match
        let p = match;
        while (p.parent) {
            if (p.parent._children) {
                p.parent.children = p.parent._children;
                p.parent._children = null;
            }
            p = p.parent;
        }
        window.treeUpdate(match);

        // Center View on Match
        // Calculate transform to center (x, y)
        // Note: x is vertical, y is horizontal in our tree
        const t = d3.zoomIdentity.translate(width / 2 - match.y, height / 2 - match.x).scale(1.5);
        svg.transition().duration(1000).call(zoom.transform, t);

        // Highlight via CSS class
        d3.selectAll('.node rect').style('stroke', '#fff'); // Reset
        d3.select(`#node-${match.id}`).style('stroke', '#facc15').style('stroke-width', '4px'); // Yellow highlight

        showDetails(match.data);
    } else {
        alert("No match found");
    }
}
