# Handwritten Digit Classification (E6)

A machine learning project that classifies handwritten digits (0-9) using multiple classification algorithms on the scikit-learn digits dataset.

## 📊 Dataset

- **Source**: scikit-learn datasets (digits dataset)
- **Size**: 1,797 samples with 64 features (8x8 pixel images)
- **Classes**: 10 (digits 0-9)
- **Train/Test Split**: 80/20 with random_state=42

## 🎯 Objective

Compare the performance of different machine learning classifiers for handwritten digit recognition and identify the most effective algorithm.

## 🏗️ Project Structure

```
E6 HandwrittenDigitClassification/
├── HandwrittenDigitClassification.py    # Main script
└── README.md                            # This file
```

## 🔧 Technologies Used

- **Libraries**: scikit-learn, numpy, pandas, matplotlib, seaborn
- **Python**: 3.7+

## 📈 Models Implemented

1. **Random Forest** - Ensemble learning with 100 decision trees
2. **Support Vector Machine (SVM)** - RBF kernel classifier
3. **K-Nearest Neighbors (KNN)** - k=5 neighbors
4. **Decision Tree** - Single decision tree classifier
5. **Logistic Regression** - Linear classification with max_iter=1000

## 🔄 Workflow

1. **Data Loading** - Load digits dataset from scikit-learn
2. **Visualization** - Display sample digits (first 10) with their labels
3. **Preprocessing**:
   - 80/20 train-test split
   - StandardScaler normalization applied to training and test sets
4. **Model Training** - Train all 5 classifiers on normalized training data
5. **Evaluation** - Calculate metrics for each model:
   - **Accuracy**: Overall correctness
   - **Precision**: Weighted precision across all classes
   - **Recall**: Weighted recall across all classes
   - **F1-Score**: Harmonic mean of precision and recall
   - **Confusion Matrix**: Visual representation of predictions vs actual
   - **Classification Report**: Detailed per-class metrics
6. **Comparison** - Compare all classifiers and generate comparison table

## 📊 Expected Output

- Sample digit visualizations
- Individual confusion matrices for each classifier
- Detailed classification reports for each model
- Comparison table with accuracy, precision, recall, and F1-scores
- Printed metrics and model performance summaries

## 🚀 Running the Project

```bash
# Navigate to the project directory
cd "E6 HandwrittenDigitClassification"

# Ensure virtual environment is activated
# On Windows PowerShell:
..\..\\.venv\Scripts\Activate.ps1

# Run the script
python HandwrittenDigitClassification.py
```

## 📋 Key Features

- **Multi-Model Comparison**: Evaluates 5 different algorithms side-by-side
- **Comprehensive Metrics**: Uses multiple evaluation metrics for thorough assessment
- **Visualization**: Confusion matrices for each model to visualize performance
- **Standard Preprocessing**: Data normalization for fair comparison
- **Detailed Reporting**: Classification reports with precision, recall, and F1-scores per digit class

## 💡 Expected Results

- All models should achieve >90% accuracy on the test set
- SVM typically performs very well on this dataset
- Random Forest usually provides excellent results with good interpretability
- Confusion matrices show which digit classes are commonly confused

## 🔍 Analysis Tips

- Compare accuracy scores to determine the best performer
- Check confusion matrices to identify which digits are frequently misclassified
- Review the classification reports for per-class performance variations
- SNS heatmaps provide clear visualization of model confusion patterns

## ✅ Success Criteria

- All models successfully train and predict
- Metrics are computed for each classifier
- Confusion matrices are visualized
- Comparison results are displayed
- Project demonstrates effective use of scikit-learn for multi-class classification
