import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from . import config

def get_cluster_stats(labels, true_labels=None, df=None):
    """
    Computes per-cluster statistics including size, purity, and dominant category.
    
    Returns:
        DataFrame with columns: cluster_id, size, pct, purity, dominant_category
    """
    cluster_counts = Counter(labels)
    stats = []
    
    for cluster_id, size in sorted(cluster_counts.items()):
        if cluster_id == -1:
            continue
            
        row = {
            'cluster_id': cluster_id,
            'size': size,
            'pct': size / len(labels) * 100
        }
        
        if true_labels is not None:
            mask = labels == cluster_id
            cluster_true = true_labels[mask] if isinstance(true_labels, np.ndarray) else true_labels.iloc[mask.nonzero()[0]]
            category_counts = Counter(cluster_true)
            dominant_cat, dominant_count = category_counts.most_common(1)[0]
            row['dominant_category'] = dominant_cat
            row['purity'] = dominant_count / size
            row['num_categories'] = len(category_counts)
        
        stats.append(row)
    
    df_stats = pd.DataFrame(stats)
    if 'purity' in df_stats.columns:
        df_stats = df_stats.sort_values('purity', ascending=True)
    
    return df_stats

def get_representative_samples(embeddings, labels, df, text_col='cluster_text', n_samples=3):
    """
    Finds the most representative (central) samples for each cluster.
    
    Returns:
        Dict mapping cluster_id -> list of sample texts
    """
    from sklearn.metrics import pairwise_distances
    
    representatives = {}
    unique_labels = sorted(set(labels) - {-1})
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_indices = np.where(mask)[0]
        
        if len(cluster_embeddings) < n_samples:
            indices = cluster_indices
        else:
            centroid = cluster_embeddings.mean(axis=0).reshape(1, -1)
            distances = pairwise_distances(cluster_embeddings, centroid).flatten()
            closest_idx = np.argsort(distances)[:n_samples]
            indices = cluster_indices[closest_idx]
        
        samples = df.iloc[indices][text_col].tolist()
        representatives[cluster_id] = samples
    
    return representatives

def get_worst_clusters(cluster_stats, n_worst=10):
    """
    Returns the worst performing clusters (lowest purity).
    """
    if 'purity' not in cluster_stats.columns:
        return cluster_stats.head(n_worst)
    
    return cluster_stats.nsmallest(n_worst, 'purity')

def get_best_clusters(cluster_stats, n_best=10):
    """
    Returns the best performing clusters (highest purity).
    """
    if 'purity' not in cluster_stats.columns:
        return cluster_stats.tail(n_best)
    
    return cluster_stats.nlargest(n_best, 'purity')

def plot_cluster_size_distribution(labels, output_path="outputs/cluster_size_distribution.png"):
    """
    Plots a histogram of cluster sizes.
    """
    cluster_counts = Counter(labels)
    if -1 in cluster_counts:
        del cluster_counts[-1]
    
    sizes = list(cluster_counts.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(sizes, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Cluster Size')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Cluster Size Distribution')
    axes[0].axvline(np.median(sizes), color='red', linestyle='--', label=f'Median: {np.median(sizes):.0f}')
    axes[0].axvline(np.mean(sizes), color='orange', linestyle='--', label=f'Mean: {np.mean(sizes):.0f}')
    axes[0].legend()
    
    axes[1].hist(sizes, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Cluster Size (log scale)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Cluster Size Distribution (Log Scale)')
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved cluster size distribution to {output_path}")

def plot_purity_distribution(cluster_stats, output_path="outputs/cluster_purity_distribution.png"):
    """
    Plots histogram of per-cluster purity scores.
    """
    if 'purity' not in cluster_stats.columns:
        print("No purity scores available")
        return
    
    purities = cluster_stats['purity'].values
    
    plt.figure(figsize=(10, 6))
    plt.hist(purities, bins=20, color='forestgreen', edgecolor='white', alpha=0.8)
    plt.xlabel('Cluster Purity')
    plt.ylabel('Number of Clusters')
    plt.title('Per-Cluster Purity Distribution')
    plt.axvline(np.median(purities), color='red', linestyle='--', label=f'Median: {np.median(purities):.2%}')
    plt.axvline(np.mean(purities), color='orange', linestyle='--', label=f'Mean: {np.mean(purities):.2%}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved purity distribution to {output_path}")

def generate_cluster_quality_report(labels, true_labels, df, embeddings, output_dir="outputs"):
    """
    Generates a complete cluster quality analysis report.
    
    Outputs:
        - cluster_quality_report.csv: Per-cluster stats
        - cluster_size_distribution.png: Size histogram
        - cluster_purity_distribution.png: Purity histogram
        - worst_clusters_analysis.csv: Detailed worst cluster analysis
    """
    print("\n--- Generating Cluster Quality Analysis ---")
    
    cluster_stats = get_cluster_stats(labels, true_labels, df)
    cluster_stats.to_csv(f"{output_dir}/cluster_quality_report.csv", index=False)
    print(f"   > Saved per-cluster stats ({len(cluster_stats)} clusters)")
    
    plot_cluster_size_distribution(labels, f"{output_dir}/cluster_size_distribution.png")
    
    if 'purity' in cluster_stats.columns:
        plot_purity_distribution(cluster_stats, f"{output_dir}/cluster_purity_distribution.png")
    
    worst = get_worst_clusters(cluster_stats)
    representatives = get_representative_samples(embeddings, labels, df, n_samples=2)
    
    worst_analysis = []
    for _, row in worst.iterrows():
        cid = int(row['cluster_id'])
        analysis = {
            'cluster_id': cid,
            'size': row['size'],
            'purity': row.get('purity', 'N/A'),
            'dominant_category': row.get('dominant_category', 'N/A'),
            'num_categories': row.get('num_categories', 'N/A'),
            'sample_1': representatives.get(cid, ['N/A'])[0][:200],
            'sample_2': representatives.get(cid, ['N/A', 'N/A'])[1][:200] if len(representatives.get(cid, [])) > 1 else 'N/A'
        }
        worst_analysis.append(analysis)
    
    pd.DataFrame(worst_analysis).to_csv(f"{output_dir}/worst_clusters_analysis.csv", index=False)
    print(f"   > Saved worst clusters analysis (top {len(worst)} problematic clusters)")
    
    summary = {
        'total_clusters': len(cluster_stats),
        'median_size': cluster_stats['size'].median(),
        'mean_size': cluster_stats['size'].mean(),
        'min_size': cluster_stats['size'].min(),
        'max_size': cluster_stats['size'].max()
    }
    
    if 'purity' in cluster_stats.columns:
        summary['median_purity'] = cluster_stats['purity'].median()
        summary['mean_purity'] = cluster_stats['purity'].mean()
        summary['clusters_above_90pct'] = (cluster_stats['purity'] >= 0.9).sum()
        summary['clusters_below_50pct'] = (cluster_stats['purity'] < 0.5).sum()
    
    print(f"   > Summary: {summary['total_clusters']} clusters, median purity: {summary.get('median_purity', 'N/A'):.2%}")
    
    return cluster_stats, summary
