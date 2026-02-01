import hdbscan
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS, Birch, BisectingKMeans, MiniBatchKMeans
import numpy as np
from . import config

def run_kmeans(embeddings, n_clusters=50, quiet=False):
    if not quiet:
        start = time.time()
        print(f"\n--- KMeans (k={n_clusters}) ---")
        print(f"   > Input data shape: {embeddings.shape}")
        print(f"   > Initializing model (n_init=10)...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10, verbose=0)
    labels = kmeans.fit_predict(embeddings)
    
    if not quiet:
        elapsed = time.time() - start
        print(f"   > KMeans completed in {elapsed:.2f} seconds.")
    return labels

def run_minibatch_kmeans(embeddings, n_clusters=100, batch_size=1024, quiet=False):
    """MiniBatchKMeans for large-scale clustering (500k+ rows)."""
    if not quiet:
        start = time.time()
        print(f"\n--- MiniBatchKMeans (k={n_clusters}, batch={batch_size}) ---")
        print(f"   > Input data shape: {embeddings.shape}")
        print(f"   > Streaming batches for scalable clustering...")
    
    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=config.RANDOM_SEED, n_init=3)
    labels = mbk.fit_predict(embeddings)
    
    if not quiet:
        elapsed = time.time() - start
        print(f"   > MiniBatchKMeans completed in {elapsed:.2f} seconds.")
    return labels

def run_dbscan(embeddings, eps=0.5, min_samples=5, quiet=False):
    if not quiet:
        start = time.time()
        print(f"\n--- DBSCAN (eps={eps}, min_samples={min_samples}) ---")
        print(f"   > Input data shape: {embeddings.shape}")
        print(f"   > Calculating pairwise distances and finding dense regions...")
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(embeddings)
    
    if not quiet:
        elapsed = time.time() - start
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"   > DBSCAN completed in {elapsed:.2f} seconds.")
        print(f"   > Found {n_clusters} clusters and {n_noise} noise points.")
    return labels

def run_hdbscan(embeddings, min_cluster_size=10, min_samples=5, quiet=False):
    if not quiet:
        start = time.time()
        print(f"\n--- HDBSCAN (min_cluster_size={min_cluster_size}) ---")
        print(f"   > Input data shape: {embeddings.shape}")
        print(f"   > Building Minimum Spanning Tree and Condensed Tree...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(embeddings)
    
    if not quiet:
        elapsed = time.time() - start
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"   > HDBSCAN completed in {elapsed:.2f} seconds.")
        print(f"   > Found {n_clusters} clusters.")
    return labels

def run_agglomerative(embeddings, n_clusters=50):
    start = time.time()
    print(f"\n--- Agglomerative Clustering (k={n_clusters}) ---")
    print(f"   > Input data shape: {embeddings.shape}")
    print(f"   > Computing linkage matrix (ward)...")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(embeddings)
    elapsed = time.time() - start
    print(f"   > Agglomerative Clustering completed in {elapsed:.2f} seconds.")
    return labels

def run_optics(embeddings, min_samples=5, max_eps=np.inf, quiet=False):
    if not quiet:
        start = time.time()
        print(f"\n--- OPTICS (min_samples={min_samples}, max_eps={max_eps}) ---")
        print(f"   > Input data shape: {embeddings.shape}")
    
    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric='euclidean', n_jobs=-1)
    labels = optics.fit_predict(embeddings)
    
    if not quiet:
        elapsed = time.time() - start
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"   > OPTICS completed in {elapsed:.2f} seconds.")
        print(f"   > Found {n_clusters} clusters and {n_noise} noise points.")
    return labels

def run_birch(embeddings, n_clusters=50, threshold=0.5, quiet=False):
    if not quiet:
        start = time.time()
        print(f"\n--- BIRCH (n_clusters={n_clusters}, threshold={threshold}) ---")
        print(f"   > Input data shape: {embeddings.shape}")
        
    birch = Birch(n_clusters=n_clusters, threshold=threshold)
    labels = birch.fit_predict(embeddings)
    
    if not quiet:
        elapsed = time.time() - start
        print(f"   > BIRCH completed in {elapsed:.2f} seconds.")
    return labels

def run_bisecting_kmeans(embeddings, n_clusters=50, quiet=False):
    if not quiet:
        start = time.time()
        print(f"\n--- Bisecting K-Means (k={n_clusters}) ---")
        print(f"   > Input data shape: {embeddings.shape}")
    
    bkmeans = BisectingKMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED)
    labels = bkmeans.fit_predict(embeddings)
    
    if not quiet:
        elapsed = time.time() - start
        print(f"   > Bisecting K-Means completed in {elapsed:.2f} seconds.")
    return labels
