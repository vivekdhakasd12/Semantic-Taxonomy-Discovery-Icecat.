import pandas as pd
import numpy as np
import sys
from src import config, data_loader, features, clustering, evaluation, visualization, tuning, analysis, supervised

def main():
    print("=== Icecat Clustering: Full Algorithm Comparison & Panel Viz ===")
    
    print("\n[1/6] Loading Data...")
    try:
        df = data_loader.load_icecat_data()
    except FileNotFoundError:
        print(f"Error: Dataset not found at {config.DATA_PATH}")
        return

    print("\n[2/6] Generating Features...")
    df = features.create_text_features(df)
    embeddings = features.generate_embeddings(df)
    
    print("\n[3/6] Reducing Dimensions (PCA=50)...")
    embeddings_low = visualization.reduce_dimensions(embeddings, method='pca', n_components=50)

    print("\n[4/6] Tuning & Running Algorithms...")
    
    y_true = df[config.LABEL_COL] if config.LABEL_COL in df.columns else None
    results_summary = {}
    all_labels = {}
    
    experiments = [
        {
            'name': 'MiniBatchKMeans',
            'algo': 'MiniBatchKMeans',
            'grid': {'n_clusters': [100, 150], 'batch_size': [2048]} 
        },
        {
            'name': 'BisectingKMeans',
            'algo': 'BisectingKMeans',
            'grid': {'n_clusters': [100]}
        },
        {
            'name': 'BIRCH',
            'algo': 'BIRCH',
            'grid': {'threshold': [0.3, 0.5], 'n_clusters': [None]}
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
        
        print(f"   > Purity: {metrics.get('purity',0):.2%}, NMI: {metrics.get('nmi',0):.2%}")
        
    if y_true is not None:
        try:
            base_metrics = supervised.run_baseline(embeddings_low, y_true)
            results_summary['Supervised Baseline'] = base_metrics
        except Exception as e:
            print(f"   > Supervised Baseline Failed: {e}")

    print("\n[5/6] Saving Report...")
    df_results = pd.DataFrame(results_summary).T
    df_results.to_csv("outputs/clustering_comparison_report.csv")
    print("Saved report to outputs/clustering_comparison_report.csv")

    print("\n[6/6] Generating Comparison Panel...")
    
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
    
    print("\n[Bonus] Generating Metrics Bar Chart & Sankey Diagram...")
    visualization.plot_metrics_comparison(results_summary, output_path="outputs/clustering_metrics_bar.png")
    
    clustering_results = {k: v for k, v in results_summary.items() if k in all_labels}
    best_algo = max(clustering_results, key=lambda k: clustering_results[k]['purity'])
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
    
    print("Done.")

if __name__ == "__main__":
    main()
