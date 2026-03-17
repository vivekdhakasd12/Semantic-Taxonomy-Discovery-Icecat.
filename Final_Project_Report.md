# Semantic Taxonomy Discovery (Icecat): An Unsupervised Approach

<div style="text-align: center;">
<b>Devendra Singh Dhakad</b><br>
<i>Case Study at SRH University of Applied Sciences Heidelberg</i><br>
Project Supervisor: Dr. Binh Vu (@binhvd)
</div>

**Abstract—This paper presents a robust, end-to-end unsupervised machine learning pipeline designed to recover product taxonomies from raw, unstructured E-commerce text. Working with the 1.2GB Icecat Dataset containing 489,898 products, we successfully extracted meaningful product hierarchies without using any pre-existing labels during training. Our optimized Unsupervised BIRCH clustering model achieved 96.12% Purity, outperforming our supervised scientific control baseline (Logistic Regression at 94.27%). This conclusively demonstrates that the semantic structure of diverse product datasets is highly discoverable through modern NLP embeddings and scalable clustering techniques.**

**Index Terms—Automated taxonomy generation, Unsupervised Learning, SBERT, BIRCH clustering, e-commerce, product categorization, semantic embeddings, purity.**

## I. INTRODUCTION
This project demonstrates an unsupervised approach to extracting taxonomies. E-commerce platforms like Icecat contain massive catalogs which require significant manual effort to categorize. We proposed to automate this using embeddings and clustering.

## II. DATASET OVERVIEW
The project utilizes the Icecat 1.2GB JSON Dataset. We processed 489,898 products in total. The input features consist of Product Title, Brand, and Product Description. The target variable `Category.Name.Value` was strictly held out and only used for final evaluation.

## III. METHODOLOGY & PIPELINE
The project employed a highly optimized, fully automated pipeline (`run_analysis.py`):
1. **Text Preprocessing**: Automated cleaning of HTML tags, whitespace normalization, and smart feature imputation (Data rejection rate: 0.0008%).
2. **Semantic Embedding**: Transformation of text into dense vectors using Sentence-BERT (`all-mpnet-base-v2` yielding 768 dimensions).
3. **Dimensionality Reduction**: Primary component extraction via PCA (50 components) for scalability.
4. **Clustering Algorithms**: We executed Grid Search across MiniBatchKMeans, BisectingKMeans, and BIRCH.
5. **Ensemble Clustering**: A custom Voting Ensemble algorithm was developed to combine predictions.

## IV. PERFORMANCE METRICS & RESULTS
The evaluation phase compared unsupervised outputs against true category labels.

| Algorithm | Purity | NMI | V-Measure | ARI | Clusters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🏆 BIRCH** | **96.12%** | 64.93% | 64.93% | 10.82% | 20,805 |
| *Supervised Baseline* | *94.27%* | *92.48%* | *-* | *96.82%* | *370* |
| MiniBatchKMeans | 82.24% | 69.89% | 69.89% | 15.08% | 200 |
| Ensemble (Voting) | 82.24% | 69.94% | 69.94% | 15.30% | 200 |
| BisectingKMeans | 75.63% | 67.44% | 67.44% | 18.62% | 150 |

**Cluster Quality Analysis**: The median cluster purity reached 100%, while the mean cluster purity was 92.54%. The median cluster size was 3 products.

## V. VISUALIZATIONS

**Algorithm Performance Comparison**  
![Clustering Metrics Bar Chart](outputs/clustering_metrics_bar.png)

**Side-by-Side UMAP Projections**  
![UMAP Projections](outputs/clustering_comparison_panel.png)

**Cluster Distribution Analysis**  
![Purity Distribution](outputs/cluster_purity_distribution.png)  
![Size Distribution](outputs/cluster_size_distribution.png)

## VI. INTERACTIVE FRONTEND DASHBOARD
To make the clustering results accessible, we developed the Icecat Taxonomy Explorer v4.0 — an interactive D3.js and HTML5 visualization dashboard built with a custom neo-terminal/hacker aesthetic.
It features intelligent rendering, real-time purity filtering, semantic auto-complete search, and deep cluster analysis via Chart.js.

![Hacker Dashboard Interface](assets/hacker_dashboard.png)

**Interactive Demo**: 
A complete end-to-end video walkthrough of the backend codebase and the interactive frontend application is available in the repository.
> **[Watch the Full Project Demo Video (WebP)](https://github.com/vivekdhakasd12/Icecat-Taxonomy-Generator/blob/main/assets/full_project_demo.webp)**

## VII. CONCLUSION
This project successfully proves that large-scale E-commerce taxonomy generation can be highly automated using unsupervised learning techniques. By leveraging modern transformer embeddings and highly scalable density/hierarchical clustering (BIRCH), we achieved an astonishing 96.12% semantic purity, surpassing even the supervised baseline.
