# Semantic Taxonomy Discovery (Icecat)

**A Thesis-Level Unsupervised Learning Project to recover Product Taxonomies from Raw Text.**

## One-Minute Summary
We applied advanced clustering algorithms (**BIRCH**, **MiniBatchKMeans**) to a 1.2GB E-commerce dataset (Icecat) containing **489,898 products** to automatically discover product hierarchies (e.g., *Laptops*, *Tablets*, *Smartphones*) without using labels.

**Key Scientific Finding**:
*   **Our Model (Unsupervised BIRCH)**: Achieved **85.1% Purity** on 500k products.
*   **Scientific Control (Supervised Logistic Regression)**: Achieved **93.9% Accuracy**.
*   **Conclusion**: Our unsupervised approach recovers **~91%** of the theoretical maximum performance, proving that the semantic structure of products is highly discoverable even without labeled training data.

---

## Dataset
The project uses the **Icecat 1.2GB JSON Dataset**.
> **Download Link**: [Google Drive Link](https://drive.google.com/file/d/13f8GHcokLVetrKNM6cFmhaMM0fVCG1NJ/view?usp=sharing)

*   **Size**: 1.2 GB (Raw), 489,898 products.
*   **Input Features**: Title, Brand, Product Description.
*   **Target**: `Category.Name.Value` (Used only for evaluation/testing, never for training).

---

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/vivekdhakasd12/Semantic-Taxonomy-Discovery-Icecat
    cd Semantic-Taxonomy-Discovery-Icecat
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
1.  **Loads Data**: Processes the full 489,898 product dataset.
2.  **Preprocesses Text**: Cleans HTML tags and applies smart imputation for missing fields.
3.  **Embeds Text**: Uses `Sentence-BERT` (all-MiniLM-L6-v2) to create dense vectors.
4.  **Reduces Dimensions**: PCA (50 components) for efficiency.
5.  **Tunes Algorithms**: Runs Grid Search on MiniBatchKMeans and BIRCH.
6.  **Runs Control**: Trains a Supervised Logistic Regression baseline for comparison.
7.  **Visualizes**: Generates Panels, Bar Charts, and Sankey Diagrams.

---

## Results and Visualization

All outputs are saved to the `outputs/` directory.

### 1. Algorithm Comparison (Panel)
See `outputs/clustering_comparison_panel.png`.
*   Shows side-by-side clustering structures.
*   **BIRCH** produces compact, well-separated clusters similar to Ground Truth.

### 2. Performance Metrics (Full Dataset: 489,898 Products)
See `outputs/clustering_comparison_report.csv`.

| Algorithm | Purity | NMI | Notes |
| :--- | :--- | :--- | :--- |
| **Supervised Baseline** | **93.9%** | - | **Scientific Upper Bound** |
| **BIRCH** | **85.1%** | **71.9%** | **Best Unsupervised Model** |
| MiniBatchKMeans (k=150) | 78.9% | 69.2% | Scalable baseline |
| Bisecting KMeans | 73.3% | 66.7% | Hierarchical baseline |

### 3. Error Analysis (Sankey Flow)
See `outputs/sankey_BIRCH.html`.
*   Interactive flow diagram showing how "True Categories" map to "Clusters".
*   Reveals that most errors are semantic ambiguities (e.g., Tablets vs Laptops).

---

## Recent Updates

### Phase 2: Full-Scale Training (2026-02-01)
Scaled the pipeline to train on the complete dataset:
*   **MiniBatchKMeans**: Added scalable clustering algorithm for 500k+ rows.
*   **Full Dataset**: Removed sampling, now processes all 489,898 products.
*   **Result**: BIRCH Purity improved from 82.2% to 85.1% (+2.9%).
*   **Supervised Baseline**: Improved from 88.4% to 93.9% (+5.5%) with more training data.

### Phase 1: Preprocessing Improvements (2026-02-01)
*   **HTML Cleaning**: Removes HTML tags from product descriptions.
*   **Smart Imputation**: Fills empty fields using fallback priority.
*   **Data Rejection**: Only 4 rows dropped out of 489,898 (0.0008%).

---

## Repository Structure
*   `src/`: Modular Python code (Clustering, Evaluation, Visualization, etc.).
*   `run_analysis.py`: Main entry point.
*   `outputs/`: Generated reports and image artifacts.

---

## Acknowledgements

**Project Supervisor**: **Dr. Binh Vu** ([@binhvd](https://github.com/binhvd))
*   For guidance on the application of unsupervised learning techniques to e-commerce taxonomies.

**Author**: Devendra Singh Dhakad
*   *Generated as part of the Case Study at SRH University of Applied Sciences Heidelberg*
