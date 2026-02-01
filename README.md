# Semantic Taxonomy Discovery (Icecat)

**A Thesis-Level Unsupervised Learning Project to recover Product Taxonomies from Raw Text.**

## One-Minute Summary
We applied advanced clustering algorithms (**BIRCH**, **OPTICS**, **HDBSCAN**) to a 1.2GB E-commerce dataset (Icecat) to automatically discover product hierarchies (e.g., *Laptops*, *Tablets*, *Smartphones*) without using labels.

**Key Scientific Finding**:
*   **Our Model (Unsupervised BIRCH)**: Achieved **82.2% Purity**.
*   **Scientific Control (Supervised Logistic Regression)**: Achieved **88.4% Accuracy**.
*   **Conclusion**: Our unsupervised approach recovers **~93%** of the theoretical maximum performance, proving that the semantic structure of products is highly discoverable even without labeled training data.

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
    git clone https://github.com/vivekdhakasd12/Semantic-Taxonomy-Discovery
    cd Semantic-Taxonomy-Discovery
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
2.  **Preprocesses Text**: Cleans HTML tags and applies smart imputation for missing fields.
3.  **Embeds Text**: Uses `Sentence-BERT` (all-MiniLM-L6-v2) to create dense vectors.
4.  **Reduces Dimensions**: PCA (50 components) for efficiency.
5.  **Tunes Algorithms**: Runs Grid Search on KMeans, HDBSCAN, OPTICS, and BIRCH.
6.  **Runs Control**: Trains a Supervised Logistic Regression baseline for comparison.
7.  **Visualizes**: Generates Panels, Bar Charts, and Sankey Diagrams.

---

## Results and Visualization

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
| **Supervised Baseline** | **88.4%** | - | - | **Scientific Upper Bound** |
| **BIRCH** | **82.2%** | **73.9%** | - | **Best Unsupervised Model** |
| KMeans (k=100) | 75.3% | 70.2% | - | Good baseline |
| Bisecting KMeans | 72.0% | 67.1% | - | Good for hierarchy |
| HDBSCAN | 67.0% | 56.3% | - | High noise sensitivity |

### 3. Error Analysis (Sankey Flow)
See `outputs/sankey_BIRCH.html`.
*   Interactive flow diagram showing how "True Categories" map to "Clusters".
*   Reveals that most errors are semantic ambiguities (e.g., Tablets vs Laptops).

---

## Recent Updates

### Phase 1: Preprocessing Improvements (2026-02-01)
Added advanced text preprocessing to improve data quality:
*   **HTML Cleaning**: Removes HTML tags (`<b>`, `<br>`, `<div>`) from product descriptions using BeautifulSoup.
*   **Smart Imputation**: Fills empty description fields using fallback priority (Description > LongDesc > Title > ProductName > Brand).
*   **Result**: Reduced rejected rows from ~0.02% to 0.002% (only 1 row dropped out of 50,000).
*   **Impact**: BIRCH Purity improved from 78.6% to 82.2% (+3.6%).

---

## Repository Structure
*   `src/`: Modular Python code (Clustering, Evaluation, Visualization, etc.).
*   `run_analysis.py`: Main entry point.
*   `outputs/`: Generated reports and image artifacts.
*   `Icecat_Clustering_Analysis.ipynb`: Notebook version of the pipeline.

---

## Acknowledgements

**Project Supervisor**: **Dr. Binh Vu** ([@binhvd](https://github.com/binhvd))
*   For guidance on the application of unsupervised learning techniques to e-commerce taxonomies.

**Author**: Devendra Singh Dhakad
*   *Generated as part of the Case Study at SRH University of Applied Sciences Heidelberg*
