# Real-World Data Project: Healthcare Disease Risk Classification

## Project Overview

This project is a complete end-to-end healthcare data science portfolio project. It analyzes the Breast Cancer Wisconsin Diagnostic dataset and builds machine learning models that classify tumors as benign or malignant using clinical measurements computed from digitized cell nuclei images.

The goal is to demonstrate a practical workflow that includes dataset preparation, data cleaning, exploratory data analysis, visualization, feature engineering, model training, model evaluation, and model export.

## Domain Description

Healthcare analytics helps clinicians and healthcare organizations identify risk patterns, prioritize patient review, and support evidence-based decision-making. In this project, diagnostic measurements are used to estimate disease risk through supervised classification.

## Dataset Details

- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** Bundled with `scikit-learn`
- **Records:** 569
- **Original predictive features:** 30
- **Target variable:** `diagnosis_label`
- **Classes:** benign and malignant

The dataset is saved locally at:

```text
dataset/breast_cancer_wisconsin.csv
```

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- Jupyter Notebook

## Project Structure

```text
real-world-data-project/
|
├── dataset/
│   └── breast_cancer_wisconsin.csv
├── images/
│   ├── boxplot_worst_area.png
│   ├── category_wise_feature_means.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── model_comparison.png
│   ├── pairplot_selected_features.png
│   ├── roc_curve.png
│   ├── scatter_radius_texture.png
│   ├── target_distribution.png
│   └── trend_radius_measurements.png
├── models/
│   └── breast_cancer_classifier.pkl
├── notebooks/
│   └── healthcare_breast_cancer_analysis.ipynb
├── main.ipynb
├── README.md
└── requirements.txt
```

## Preprocessing Steps

- Checked and handled missing values.
- Removed duplicate rows.
- Detected outliers with the IQR method.
- Capped extreme numeric values to reduce outlier influence.
- Added engineered features:
  - `area_perimeter_ratio`
  - `compactness_ratio`
  - `mean_texture_smoothness`
  - `worst_area_log`
- Encoded the diagnosis target variable.
- Scaled numerical features inside model pipelines.
- Split the data into training and testing sets with stratification.

## Exploratory Data Analysis

The notebook includes:

- Summary statistics
- Class distribution analysis
- Correlation analysis
- Distribution analysis
- Category-wise feature comparison
- Trend analysis across ordered samples
- Pairwise relationships between selected measurements

## Key EDA Findings

- Malignant cases generally show larger radius, perimeter, area, concavity, and worst-area values.
- Several size-related variables are strongly correlated, especially radius, perimeter, and area.
- Worst-case measurements tend to separate benign and malignant classes more clearly than some mean measurements.
- The classes are not perfectly balanced, so stratified splitting and multiple classification metrics are useful.

## Machine Learning Models

Two classification models were trained and compared:

- Logistic Regression
- Random Forest Classifier

The best-performing model was saved as:

```text
models/breast_cancer_classifier.pkl
```

## Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | MSE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9983 | 0.0263 |
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9964 | 0.0351 |


## Visualizations

All generated plots are saved in the `images/` folder. The project includes bar charts, line charts, histograms, boxplots, scatter plots, pairplots, heatmaps, a confusion matrix, a ROC curve, and a model comparison chart.

Example visual outputs:

![Correlation Heatmap](images/correlation_heatmap.png)

![ROC Curve](images/roc_curve.png)

![Model Comparison](images/model_comparison.png)

## Conclusions and Insights

- The analysis shows that tumor size, concavity, and worst-case measurements are highly informative for classification.
- The best model was **Random Forest**, achieving an accuracy of **0.9737**, F1-score of **0.9630**, and ROC-AUC of **0.9983** on the test set.
- The model can support healthcare decision-making by helping flag high-risk diagnostic profiles for further clinical review.
- This project is suitable for internships, academic submissions, resume projects, and GitHub portfolio presentation.

## Limitations

- The dataset is relatively small compared with modern hospital-scale patient records.
- Features are derived from image measurements, not full patient histories.
- The model should not be used as a standalone medical diagnostic tool.

## Future Improvements

- Test additional algorithms such as XGBoost or support vector machines.
- Add cross-validation and hyperparameter tuning.
- Build a Streamlit dashboard for interactive predictions.
- Incorporate model explainability with SHAP or permutation importance.
- Validate the model on a larger external clinical dataset.

## How to Run

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run:

```text
main.ipynb
```

## Generated Run Summary

- Dataset shape after duplicate removal: 569 rows x 36 columns
- Duplicate rows removed: 0
- Missing values found: 0
- Best model: Random Forest
- Best test accuracy: 0.9737
- Best test F1-score: 0.9630
- Best ROC-AUC: 0.9983
