
import re
import json
import pandas as pd
import numpy as np

def extract_and_process_sankey_data(html_path, output_path):
    print(f"Reading {html_path}...")
    with open(html_path, 'r') as f:
        content = f.read()
    
    # Look for the data payload. It's usually inside a script tag calling Plotly.newPlot
    # Pattern: Plotly.newPlot(..., [data], ...)
    
    # We'll look for the 'node' and 'link' dictionaries
    print("Extracting data tables...")
    
    # Simple regex to find the labels
    labels_match = re.search(r'"label":\s*(\[[^\]]+\])', content)
    if not labels_match:
        print("Error: Could not find node labels")
        return

    labels = json.loads(labels_match.group(1))
    print(f"Found {len(labels)} labels")
    
    # Regex for source, target, value
    source_match = re.search(r'"source":\s*(\[[^\]]+\])', content)
    target_match = re.search(r'"target":\s*(\[[^\]]+\])', content)
    value_match = re.search(r'"value":\s*(\[[^\]]+\])', content)
    
    if not (source_match and target_match and value_match):
        print("Error: Could not find link data")
        return
        
    source = json.loads(source_match.group(1))
    target = json.loads(target_match.group(1))
    value = json.loads(value_match.group(1))
    
    print(f"Found {len(source)} links")
    
    # Reconstruct the mapping
    # In Sankey: Source is usually valid Category, Target is usually Cluster (or vice versa depending on flow)
    # Based on my previous code, it's True Category -> Cluster.
    
    # Let's map indices to labels
    # Source indices point to Category Names
    # Target indices point to Cluster Names/IDs
    
    cluster_composition = {}
    
    for s_idx, t_idx, v in zip(source, target, value):
        cat_name = labels[s_idx]
        cluster_name = labels[t_idx]
        
        # Check if 'Cluster' is in the target name to confirm direction
        # If the flow is Cluster -> Category, then source is Cluster. Use logic to detect.
        
        if "Cluster" in cluster_name and "Cluster" not in cat_name:
             # Category -> Cluster flow
            final_cluster_id = cluster_name.replace("Cluster ", "").strip()
            final_cat = cat_name
        elif "Cluster" in cat_name and "Cluster" not in cluster_name:
             # Cluster -> Category flow (unlikely for taxonomy discovery, but possible if visualized that way)
            final_cluster_id = cat_name.replace("Cluster ", "").strip()
            final_cat = cluster_name
        else:
            # Fallback or mixed level
            continue
            
        if final_cluster_id not in cluster_composition:
            cluster_composition[final_cluster_id] = []
            
        cluster_composition[final_cluster_id].append({
            "category": final_cat,
            "count": v
        })
    
    # Process into rich data format
    rich_data = {}
    
    for cid, items in cluster_composition.items():
        total_count = sum(item['count'] for item in items)
        
        # Sort by count desc
        sorted_items = sorted(items, key=lambda x: x['count'], reverse=True)
        
        composition = []
        for item in sorted_items:
            composition.append({
                "name": item['category'],
                "count": item['count'],
                "percentage": round(item['count'] / total_count * 100, 1)
            })
            
        # Get dominant
        dominant = composition[0]
        
        rich_data[cid] = {
            "breakdown": composition,  # Full list (or top N)
            "total_size": total_count,
            "purity": dominant['count'] / total_count,
            "dominant_category": dominant['name']
        }

    print(f"Processed {len(rich_data)} clusters")
    
    # Merge with existing cluster_data.json if needed, or just save this
    # We'll save this as a separate detailed file
    with open(output_path, 'w') as f:
        json.dump(rich_data, f, indent=2)
    print(f"Saved rich data to {output_path}")

if __name__ == "__main__":
    extract_and_process_sankey_data(
        "outputs/sankey_BIRCH.html",
        "outputs/cluster_data_rich.json"
    )
