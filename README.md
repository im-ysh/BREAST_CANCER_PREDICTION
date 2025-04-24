# CANCER_PREDICTION
Cancer prediction project using basic neural network

---

# 🧬 Cancer Prediction using Neural Networks

This project showcases a **binary classification model** built using a custom **Neural Network in PyTorch** to predict whether a tumor is **Benign (0)** or **Malignant (1)**. The dataset is derived from diagnostic measurements in medical reports.

---

## 📁 Dataset

The dataset is a tabular CSV file with numeric features representing medical metrics extracted from cell nuclei images.

### 🔹 Structure:
- Feature Columns: Continuous variables (e.g., radius, texture)
- Target Column: Diagnosis with labels `'B'` (Benign) and `'M'` (Malignant)

### 🏷️ Class Labels
```python
{'B': 0, 'M': 1}
```

---

## 🧠 Model Architecture

The neural network is implemented using `torch.nn.Module` and includes:

- **Input Layer**: Matches the number of features
- **2 Hidden Layers**: Each with ReLU activation
- **Output Layer**: Single neuron with Sigmoid activation (for binary classification)
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam

---

## 🔧 Hyperparameters

| Parameter         | Value             |
|------------------|-------------------|
| Batch Size       | 32                |
| Learning Rate    | 0.001             |
| Epochs           | 50                |
| Optimizer        | Adam              |
| Loss Function    | BCEWithLogitsLoss |

---

## 🔄 Workflow

### 📦 Preprocessing
- Dropped unnecessary columns: `id`, `Unnamed: 32`
- Converted labels: `'B' → 0`, `'M' → 1`
- Split data into training and testing sets
- Converted data into PyTorch tensors

### 🚂 Training
- Model trained over 50 epochs
- Tracked training loss at each epoch

### 🧪 Evaluation
- Used `accuracy_score` from `sklearn` for final performance
- Confusion matrix plotted using `seaborn`

```python
Accuracy of the model on the test data: ~96%
```

---

## 📊 Visualization

Includes:
- Count plot of label distribution
- Heatmap of feature correlations
- Accuracy and loss plots
- Confusion matrix heatmap

Example:
```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
```

---

## 📦 Libraries Used

| Library        | Purpose                                  |
|----------------|------------------------------------------|
| `pandas`       | Data loading and manipulation            |
| `numpy`        | Numerical operations                     |
| `matplotlib`   | Data visualization                       |
| `seaborn`      | Statistical plots and heatmaps           |
| `torch`        | Building and training the neural network |
| `sklearn`      | Metrics and data preprocessing           |

---

---

## ▶️ How to Run

1. Clone or download the repository
2. Place `data_day1.csv` in the project directory
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn torch scikit-learn
   ```
4. Open `GEN_AI_DAY_.ipynb` in Jupyter Notebook or Google Colab
5. Run all cells sequentially

---

## 🏆 Results

- **Test Accuracy Achieved**: ~96%
- **Dataset Size**: ~569 samples
- High classification performance on medical diagnostic data

---

## 🙏 Acknowledgements

- PyTorch for model implementation
- Matplotlib & Seaborn for visualization
- Scikit-learn for metrics and preprocessing
- Data inspired by Breast Cancer Wisconsin Dataset

---

