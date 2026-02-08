import pandas as pd
import numpy as np
import sys
import gc
from src import config, data_loader, features, clustering, evaluation, visualization, tuning, analysis, supervised
from src import cluster_analysis, ensemble

def main():
    print("=== Icecat Clustering: Maximum Output Analysis ===")
    print(f"   Embedding Model: {config.EMBEDDING_MODEL}")
    
    print("\n[1/8] Loading Data...")
    try:
        df = data_loader.load_icecat_data()
    except FileNotFoundError:
        print(f"Error: Dataset not found at {config.DATA_PATH}")
        return

    print("\n[2/8] Generating Features...")
    df = features.create_text_features(df)
    embeddings = features.generate_embeddings(df)
    
    print("\n[3/8] Reducing Dimensions (PCA=50)...")
    embeddings_low = visualization.reduce_dimensions(embeddings, method='pca', n_components=50)

    print("\n[4/8] Tuning & Running Algorithms...")
    
    y_true = df[config.LABEL_COL] if config.LABEL_COL in df.columns else None
    results_summary = {}
    all_labels = {}
    
    experiments = [
        {
            'name': 'MiniBatchKMeans',
            'algo': 'MiniBatchKMeans',
            'grid': {'n_clusters': [80, 100, 120, 150, 200], 'batch_size': [1024, 2048]} 
        },
        {
            'name': 'BisectingKMeans',
            'algo': 'BisectingKMeans',
            'grid': {'n_clusters': [80, 100, 150]}
        },
        {
            'name': 'BIRCH',
            'algo': 'BIRCH',
            'grid': {'threshold': [0.3, 0.5], 'n_clusters': [None]}  # Reduced for memory
        }
    ]
    
    for exp in experiments:
        algo_name = exp['name']
        algo_code = exp['algo']
        grid = exp['grid']
        
        best_params, best_score, _ = tuning.tune_hyperparameters(
            embeddings_low, algo_code, grid, sample_size=10000
        )
        
        print(f"   > Final run for {algo_name} using {best_params}...")
        
        if algo_code == 'MiniBatchKMeans':
            labels = clustering.run_minibatch_kmeans(embeddings_low, **best_params, quiet=True)
        elif algo_code == 'BisectingKMeans':
            labels = clustering.run_bisecting_kmeans(embeddings_low, **best_params, quiet=True)
        elif algo_code == 'BIRCH':
            labels = clustering.run_birch(embeddings_low, **best_params, quiet=True)
        else:
            labels = np.zeros(len(embeddings_low))

        metrics = evaluation.compute_metrics(embeddings_low, labels, y_true)
        metrics['best_params'] = str(best_params)
        results_summary[algo_name] = metrics
        all_labels[algo_name] = labels
        
        print(f"   > Purity: {metrics.get('purity',0):.2%}, NMI: {metrics.get('nmi',0):.2%}, V-Measure: {metrics.get('v_measure',0):.2%}")
        
        # Free memory after each experiment
        gc.collect()

    print("\n[5/8] Running Ensemble Clustering...")
    try:
        ensemble_labels = ensemble.run_ensemble(all_labels, method='voting', n_clusters=100)
        ensemble_metrics = evaluation.compute_metrics(embeddings_low, ensemble_labels, y_true)
        results_summary['Ensemble (Voting)'] = ensemble_metrics
        all_labels['Ensemble (Voting)'] = ensemble_labels
        print(f"   > Ensemble Purity: {ensemble_metrics.get('purity',0):.2%}, NMI: {ensemble_metrics.get('nmi',0):.2%}")
    except Exception as e:
        print(f"   > Ensemble failed: {e}")
        
    if y_true is not None:
        try:
            base_metrics = supervised.run_baseline(embeddings_low, y_true)
            results_summary['Supervised Baseline'] = base_metrics
        except Exception as e:
            print(f"   > Supervised Baseline Failed: {e}")

    print("\n[6/8] Saving Report...")
    df_results = pd.DataFrame(results_summary).T
    
    metric_cols = ['purity', 'nmi', 'ari', 'v_measure', 'fowlkes_mallows', 'homogeneity', 'completeness', 'silhouette', 'n_clusters']
    existing_cols = [c for c in metric_cols if c in df_results.columns]
    df_results = df_results[existing_cols + [c for c in df_results.columns if c not in existing_cols]]
    
    df_results.to_csv("outputs/clustering_comparison_report.csv")
    print("Saved report to outputs/clustering_comparison_report.csv")
    
    print("\n   Extended Metrics Summary:")
    print("-" * 80)
    for algo, metrics in results_summary.items():
        if 'purity' in metrics:
            print(f"   {algo:25} | Purity: {metrics['purity']:.2%} | NMI: {metrics.get('nmi',0):.2%} | V-Measure: {metrics.get('v_measure',0):.2%} | ARI: {metrics.get('ari',0):.2%}")
    print("-" * 80)

    print("\n[7/8] Cluster Quality Analysis...")
    clustering_results = {k: v for k, v in results_summary.items() if k in all_labels and 'purity' in v}
    best_algo = max(clustering_results, key=lambda k: clustering_results[k]['purity'])
    print(f"   > Analyzing best model: {best_algo}")
    
    cluster_stats, summary = cluster_analysis.generate_cluster_quality_report(
        all_labels[best_algo], 
        y_true, 
        df, 
        embeddings_low
    )

    print("\n[8/8] Generating Visualizations...")
    
    print("   > Calculating UMAP 2D projection (this may take a moment)...")
    if len(embeddings) > 10000:
        idx = np.random.choice(len(embeddings), 10000, replace=False)
        emb_viz = embeddings_low[idx]
        true_viz = y_true.iloc[idx] if y_true is not None else None
        labels_viz = {k: v[idx] for k, v in all_labels.items()}
    else:
        emb_viz = embeddings_low
        true_viz = y_true
        labels_viz = all_labels
        
    embeddings_2d = visualization.reduce_dimensions(emb_viz, method='umap', n_components=2)
    
    visualization.plot_comparison_panel(
        embeddings_2d, 
        labels_viz, 
        true_labels=true_viz, 
        output_path="outputs/clustering_comparison_panel.png"
    )
    
    print("   > Generating Metrics Bar Chart & Sankey Diagram...")
    visualization.plot_metrics_comparison(results_summary, output_path="outputs/clustering_metrics_bar.png")
    
    print(f"   > Generating Sankey for Best Clustering Model: {best_algo}")
    
    df_sankey = pd.DataFrame({
        'true_label': y_true,
        'cluster': all_labels[best_algo]
    })
    
    visualization.plot_sankey_flow(
        df_sankey, 
        true_col='true_label', 
        pred_col='cluster', 
        output_path=f"outputs/sankey_{best_algo}.html",
        title=f"Flow: True Categories -> {best_algo} Clusters"
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Best Model: {best_algo} with Purity: {results_summary[best_algo]['purity']:.2%}")
    print(f"\nOutputs saved to outputs/ directory:")
    print("  - clustering_comparison_report.csv (all metrics)")
    print("  - cluster_quality_report.csv (per-cluster stats)")
    print("  - cluster_size_distribution.png")
    print("  - cluster_purity_distribution.png")
    print("  - worst_clusters_analysis.csv")
    print("  - clustering_comparison_panel.png")
    print("  - clustering_metrics_bar.png")
    print(f"  - sankey_{best_algo}.html")
    print("="*60)

if __name__ == "__main__":
    main()

