# 🩸 Automated Blood Anomaly Detection

[](https://www.google.com/search?q=https://www.python.org/)
[](https://www.google.com/search?q=https://xgboost.readthedocs.io/)
[](https://www.google.com/search?q=%23-final-results)

This repository contains an end-to-end Machine Learning pipeline for detecting hematological anomalies (Leukemia, Anemia, and Infections) from blood cell morphological features. By utilizing **Recursive Feature Elimination (RFECV)** and **Hyperparameter Tuning**, the model identifies critical biomarkers with high precision.

-----

## 📊 Key Insights & Visualizations

### 1\. Feature Distribution (Normal vs. Anomaly)

Anomalous cells typically exhibit a shift in diameter. While normal cells peak between 7–10 μm, anomalies often appear as larger "Blast" cells or smaller fragmented cells.

><img width="1916" height="895" alt="image" src="https://github.com/user-attachments/assets/923b7160-f24d-4002-87e3-8883627bc354" />


### 2\. The "Elbow" of Feature Selection

We reduced the complexity of the model from 24 features down to the **11 most impactful markers** (such as Chromatin Density and Nucleus Area %) to prevent overfitting and improve speed.

> <img width="1919" height="690" alt="image" src="https://github.com/user-attachments/assets/92743dc5-6058-4a0f-b16c-a11138c60dc6" />


### 3\. Model Confusion Matrix

The final tuned XGBoost model shows exceptional performance in distinguishing between Normal (0) and Anomaly (1) classes, achieving a recall of **92.73%**.

><img width="1916" height="961" alt="image" src="https://github.com/user-attachments/assets/f5a14265-2d34-4579-8538-a91cc45849b4" />


-----

## 🛠️ Technical Workflow

### 1\. Data Preprocessing

  * **Outlier Detection:** Used **Local Outlier Factor (LOF)** to remove 294 noisy samples (5% contamination).
  * **Scaling:** Applied `StandardScaler` to normalize feature distributions for balanced weightage.
  * **Normality Testing:** Conducted Shapiro-Wilk and D'Agostino’s K² tests to analyze feature skewness.

### 2\. Feature Engineering

Used **RFECV** to identify the top 11 features. Key drivers included:

  * `chromatin_density`
  * `nucleus_area_pct`
  * `cell_diameter_um`
  * `granularity_score`

### 3\. Final Model: Tuned XGBoost

The model was optimized using `RandomizedSearchCV` with the following parameters:

  * **Learning Rate:** 0.2
  * **Max Depth:** 5
  * **N-Estimators:** 200
  * **Scale Pos Weight:** 3.37 (to handle class imbalance)

-----

## 🏆 Final Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **96.60%** |
| **Recall (Anomalies)** | **92.73%** |
| **F1-Score** | **94.38%** |

-----

## 📂 Project Structure

  * `blood-anomaly-detection (4).ipynb`: Full data science lifecycle (EDA to Export).
  * `model.pkl`: The final trained XGBoost model.
  * `scaler.pkl`: Pre-processing object for new data.
  * `model_ffs.pkl`: Alternative model using Forward Feature Selection.

## 🚀 How to Run

1.  Clone this repository.
2.  Install dependencies: `pip install xgboost scikit-learn pandas seaborn`.
3.  Load the model using `pickle`:

<!-- end list -->

```python
import pickle
model = pickle.load(open("model.pkl", "rb"))
```

-----

**Author:** [9022574361](https://www.google.com/search?q=https://github.com/9022574361)  
**Project:** Blood Anomaly Detection System (Machine Learning)

-----

### 💡 Important Note on Images:

To make these images show up on GitHub, you **must** run these lines in your Jupyter Notebook before you push to Git:

```python
# Run these in separate cells to save your images
plt.figure(47); plt.savefig('diameter_dist.png')
plt.figure(124); plt.savefig('rfecv_curve.png')
plt.figure(127); plt.savefig('confusion_matrix.png')

# Then run your git commands again
!git add .
!git commit -m "Added README and project plots"
!git push origin main
```
