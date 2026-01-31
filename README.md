# Icecat Taxonomy Discovery & Clustering 

**A Thesis-Level Unsupervised Learning Project to recover Product Taxonomies from Raw Text.**

## One-Minute Summary
We applied advanced clustering algorithms (**BIRCH**, **OPTICS**, **HDBSCAN**) to a 1.2GB E-commerce dataset (Icecat) to automatically discover product hierarchies (e.g., *Laptops*, *Tablets*, *Smartphones*) without using labels.

**Key Scientific Finding**:
*   **Our Model (Unsupervised BIRCH)**: Achieved **78.6% Purity**.
*   **Scientific Control (Supervised Logistic Regression)**: Achieved **86.8% Accuracy**.
*   **Conclusion**: Our unsupervised approach recovers **~90%** of the theoretical maximum performance, proving that the semantic structure of products is highly discoverable even without labeled training data.

---

## Dataset
The project uses the **Icecat 1.2GB JSON Dataset**.
> **Download Link**: [Google Drive Link](https://drive.google.com/file/d/13f8GHcokLVetrKNM6cFmhaMM0fVCG1NJ/view?usp=sharing)

*   **Size**: 1.2 GB (Raw), 500k+ products.
*   **Input Features**: Title, Brand, Product Description.
*   **Target**: `Category.Name.Value` (Used only for evaluation/testing, never for training).

---

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/vivekdhakasd12/Icecat-Taxonomy-Generator
    cd Icecat-Taxonomy-Generator
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

We optimized the entire pipeline into a single, robust CLI script.

**Run the Full Analysis**:
```bash
python run_analysis.py
```

**What happens?**
1.  **Loads Data**: Efficiently samples 50k rows from the large JSON.
2.  **Embeds Text**: Uses `Sentence-BERT` (all-MiniLM-L6-v2) to create dense vectors.
3.  **Reduces Dimensions**: PCA (50 components) for efficiency.
4.  **Tunes Algorithms**: Runs Grid Search on KMeans, HDBSCAN, OPTICS, and BIRCH.
5.  **Runs Control**: Trains a Supervised Logistic Regression baseline for comparison.
6.  **Visualizes**: Generates Panels, Bar Charts, and Sankey Diagrams.

---

## Results & Visualization

All outputs are saved to the `outputs/` directory.

### 1. Algorithm Comparison (Panel)
See `outputs/clustering_comparison_panel.png`.
*   Shows side-by-side clustering structures.
*   **BIRCH** produces compact, well-separated clusters similar to Ground Truth.
*   **OPTICS/HDBSCAN** struggle with density variations in high-dimensional text space.

### 2. Performance Metrics
See `outputs/clustering_comparison_report.csv`.

| Algorithm | Purity | NMI | ARI | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Supervised Baseline** | **86.8%** | - | - | **Scientific Upper Bound** |
| **BIRCH** | **78.6%** | **68.6%** | **21.7%** | **Best Unsupervised Model** |
| KMeans ($k=100$) | 72.3% | 66.8% | 20.9% | Good baseline |
| Bisecting KMeans | 69.3% | 64.3% | 22.1% | Good for hierarchy |
| HDBSCAN | 65.4% | 54.7% | 2.2% | High noise sensitivity |

### 3. Error Analysis (Sankey Flow)
See `outputs/sankey_BIRCH.html`.
*   Interactive flow diagram showing how "True Categories" map to "Clusters".
*   Reveals that most errors are semantic ambiguities (e.g., Tablets vs Laptops).

---

## Repository Structure
*   `src/`: Modular Python code (Clustering, Evaluation, Visualization, etc.).
*   `run_analysis.py`: Main entry point.
*   `outputs/`: Generated reports and image artifacts.
*   `Icecat_Clustering_Analysis.ipynb`: Notebook version of the pipeline.

---

## 🎓 Acknowledgements

**Project Supervisor**: **Dr. Binh Vu** ([@binhvd](https://github.com/binhvd))
*   For guidance on the application of unsupervised learning techniques to e-commerce taxonomies.

**Author**: Devendra Singh Dhakad
*   *Generated as part of the Master's Thesis/Project at [University Name]*
