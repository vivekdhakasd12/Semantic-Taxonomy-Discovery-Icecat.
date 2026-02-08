import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

class ConsensusEnsemble:
    """
    Consensus Clustering via Co-Association Matrix.
    
    Builds a similarity matrix based on how often pairs of points 
    are clustered together across multiple clustering solutions.
    """
    
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.labels_ = None
        
    def fit_predict(self, labels_dict):
        """
        Args:
            labels_dict: Dict mapping algorithm name -> cluster labels array
            
        Returns:
            Consensus cluster labels
        """
        n_samples = len(next(iter(labels_dict.values())))
        n_algorithms = len(labels_dict)
        
        print(f"   > Building co-association matrix from {n_algorithms} algorithms...")
        
        co_assoc = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        for algo_name, labels in labels_dict.items():
            labels = np.array(labels)
            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    continue
                mask = labels == cluster_id
                indices = np.where(mask)[0]
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        co_assoc[indices[i], indices[j]] += 1
                        co_assoc[indices[j], indices[i]] += 1
        
        co_assoc /= n_algorithms
        
        dissimilarity = 1 - co_assoc
        np.fill_diagonal(dissimilarity, 0)
        
        print(f"   > Running hierarchical clustering on consensus matrix...")
        
        agg = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed',
            linkage='average'
        )
        self.labels_ = agg.fit_predict(dissimilarity)
        
        return self.labels_


class VotingEnsemble:
    """
    Majority Voting Ensemble for clustering.
    
    For each pair of points, counts how many algorithms agree they 
    belong to the same cluster, then uses this to form final clusters.
    """
    
    def __init__(self, n_clusters=100, method='hungarian'):
        self.n_clusters = n_clusters
        self.method = method
        self.labels_ = None
        
    def fit_predict(self, labels_dict):
        """
        Uses label alignment to combine multiple clustering solutions.
        
        Args:
            labels_dict: Dict mapping algorithm name -> cluster labels array
            
        Returns:
            Combined cluster labels using majority voting
        """
        labels_list = list(labels_dict.values())
        n_samples = len(labels_list[0])
        
        aligned = self._align_labels(labels_list)
        
        print(f"   > Performing majority voting across {len(labels_list)} algorithms...")
        
        final_labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = [aligned[j][i] for j in range(len(aligned))]
            votes = [v for v in votes if v != -1]
            if votes:
                final_labels[i] = Counter(votes).most_common(1)[0][0]
            else:
                final_labels[i] = -1
        
        self.labels_ = final_labels
        return self.labels_
    
    def _align_labels(self, labels_list):
        """
        Aligns cluster labels across multiple clustering solutions 
        using the first solution as reference.
        """
        from scipy.optimize import linear_sum_assignment
        
        reference = np.array(labels_list[0])
        aligned = [reference]
        
        for labels in labels_list[1:]:
            labels = np.array(labels)
            
            ref_unique = set(reference[reference != -1])
            lab_unique = set(labels[labels != -1])
            
            if not ref_unique or not lab_unique:
                aligned.append(labels)
                continue
            
            max_label = max(max(ref_unique), max(lab_unique)) + 1
            cost_matrix = np.zeros((max_label, max_label))
            
            for i in range(len(reference)):
                if reference[i] != -1 and labels[i] != -1:
                    cost_matrix[reference[i], labels[i]] += 1
            
            cost_matrix = cost_matrix.max() - cost_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            mapping = {old: new for new, old in zip(row_ind, col_ind)}
            
            aligned_labels = np.array([mapping.get(l, l) if l != -1 else -1 for l in labels])
            aligned.append(aligned_labels)
        
        return aligned


def run_ensemble(labels_dict, method='consensus', n_clusters=100):
    """
    Runs ensemble clustering on multiple clustering solutions.
    
    Args:
        labels_dict: Dict mapping algorithm name -> cluster labels
        method: 'consensus' or 'voting'
        n_clusters: Number of final clusters (for consensus method)
        
    Returns:
        Ensemble cluster labels
    """
    print(f"\n--- Ensemble Clustering ({method}) ---")
    
    if method == 'consensus':
        if len(next(iter(labels_dict.values()))) > 20000:
            print("   > WARNING: Consensus method is O(N^2), sampling for large datasets...")
            return _run_consensus_sampled(labels_dict, n_clusters)
        
        ensemble = ConsensusEnsemble(n_clusters=n_clusters)
        return ensemble.fit_predict(labels_dict)
    
    elif method == 'voting':
        ensemble = VotingEnsemble(n_clusters=n_clusters)
        return ensemble.fit_predict(labels_dict)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _run_consensus_sampled(labels_dict, n_clusters, sample_size=15000):
    """
    Runs consensus on a sample, then assigns remaining points to nearest cluster.
    """
    n_samples = len(next(iter(labels_dict.values())))
    idx = np.random.choice(n_samples, min(sample_size, n_samples), replace=False)
    
    sampled_labels = {k: v[idx] for k, v in labels_dict.items()}
    
    ensemble = ConsensusEnsemble(n_clusters=n_clusters)
    sample_result = ensemble.fit_predict(sampled_labels)
    
    full_labels = np.full(n_samples, -1)
    full_labels[idx] = sample_result
    
    for i in range(n_samples):
        if full_labels[i] == -1:
            votes = [labels_dict[k][i] for k in labels_dict]
            votes = [v for v in votes if v != -1]
            if votes:
                full_labels[i] = Counter(votes).most_common(1)[0][0] % n_clusters
    
    return full_labels
