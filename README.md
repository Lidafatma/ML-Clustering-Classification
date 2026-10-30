# Bank Customer Segmentation: Clustering & Classification

Customer segmentation on bank transaction data using **K-Means clustering**, followed by a **classification** step that predicts the resulting segments. Built as the final submission for Dicoding's *Machine Learning untuk Pemula* course.

## Objective
- Group bank customers into segments with similar transaction behavior (unsupervised learning).
- Train classifiers to predict the segment of a customer from their data (supervised learning).
- Give business teams a basis for targeted promotions, loyalty programs, and service improvements.

## Dataset
- Bank transaction data, **2,537 rows x 16 columns** (account, transaction amount/type/channel/date, location, customer age/occupation, transaction duration, login attempts, account balance, previous transaction date).
- Contains missing values (roughly 18-30 per column) and 21 duplicate rows.

## Repository Contents
| File | Description |
|---|---|
| `Clustering_Submission_Akhir_BMLP.ipynb` | EDA, preprocessing, K-Means clustering, cluster interpretation |
| `Klasifikasi_Submission_Akhir_BMLP.ipynb` | Classification of cluster labels, model comparison, hyperparameter tuning |

## Tools
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Yellowbrick, joblib, Google Colab

## Methodology

### 1. Clustering
1. **EDA:** correlation heatmap and histograms of numeric features.
2. **Cleaning:** removed duplicates and ID columns (TransactionID, AccountID, DeviceID, MerchantID), imputed missing values (median for numeric, mode for categorical).
3. **Outliers:** capped using the IQR method.
4. **Feature engineering:** quantile binning of `TransactionAmount` and `CustomerAge`.
5. **Scaling and encoding:** `MinMaxScaler` for numeric features, `pd.get_dummies` for categorical features.
6. **Dimensionality reduction:** PCA to 2 components.
7. **Model:** K-Means, with the number of clusters chosen using the Elbow Method (`KElbowVisualizer`) -> **k = 4**.
8. **Evaluation:** Silhouette Score = **0.5862** (computed on the 2-component PCA space).

### 2. Classification
- Target: the cluster label (`Target`) from the clustering step.
- Split: 70% train / 30% test, stratified (`random_state=42`).
- Models: Decision Tree, Random Forest, Logistic Regression.
- Tuning: `GridSearchCV` (5-fold) on Logistic Regression, best parameters `C=10`, `solver='liblinear'`.

## Results

### Customer Segments (k = 4)
| Cluster | Customers | Avg. Balance | Avg. Age | Avg. Amount | Typical Channel | Typical Occupation |
|---|---|---|---|---|---|---|
| 0 | 1,193 | 6,282 | 52.3 | 281 | ATM | Doctor |
| 1 | 231 | 1,652 | 23.7 | 277 | Branch | Student |
| 2 | 664 | 6,476 | 52.2 | 278 | Branch | Doctor |
| 3 | 449 | 1,752 | 24.3 | 301 | ATM | Student |

**Takeaway:** customers split mainly along two lines: **older, higher-balance** vs **younger, lower-balance**, and then by **preferred channel** (ATM vs Branch). Transaction amount and duration are similar across segments.

### Classification
| Model | Test Accuracy |
|---|---|
| Decision Tree | 1.0000 |
| Logistic Regression | 0.9948 |
| Logistic Regression (tuned) | 0.9961 |

## Limitations and Next Steps
Being upfront about these, since they affect how the results should be read:

- **Very high accuracy is expected, not a sign of a strong model.** The classification target is the cluster label, which was produced from the same features used for prediction, so the classifiers are largely re-learning the K-Means rules.
- **Feature space is much larger than intended.** Some ID-like columns were one-hot encoded before being dropped, producing ~7,242 features, and the 2-component PCA explains only ~7% of the variance. The silhouette score should therefore be interpreted with caution.
- **Next steps:** drop ID columns before encoding, use only meaningful features (balance, age, amount, duration, channel, occupation), retest the number of clusters, and evaluate the silhouette score in the full feature space.

## How to Run
1. Open the notebooks in Google Colab (or Jupyter).
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn yellowbrick joblib`
3. Load the dataset (update the file path in the notebook).
4. Run the clustering notebook first. It exports the clustered data used by the classification notebook.

## Author
**Ade Ziyana Walidah Fatma**
Mathematics Education graduate | Python, SQL, Data Visualization
[LinkedIn](https://www.linkedin.com/in/ade-ziyana-walidah-fatma)
