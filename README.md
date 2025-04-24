# CANCER_PREDICTION
Cancer prediction project using basic neural network

---

# 📊 CANCER_PREDICTION - Data Classification Project

This project is a machine learning workflow designed for binary classification using tabular data. The goal is to  distinguish between two classes (B → 0 and M → 1),  related to medical [ binary ]  outcome data, leveraging various data analysis and deep learning techniques.

## 🧠 Overview

This involves the following steps:
- Data loading and exploration
- Data visualization using `matplotlib` and `seaborn`
- Binary label encoding
- Model creation and training using PyTorch
- Performance evaluation and prediction

## 🧰 Libraries Used

- **Pandas** - Data manipulation
- **NumPy** - Numerical computation
- **Matplotlib / Seaborn** - Visualization
- **Torch** - Deep learning model development and training

## 📁 Dataset

The dataset used is loaded from a CSV file:

```python
data = pd.read_csv("data_day1.csv")
```

It is expected to have a binary classification target column with labels originally encoded as 'B' and 'M'.

## 📈 Features and Model Workflow

1. **Exploratory Data Analysis (EDA):**
   - View of dataset using `head()`
   - Visualizations such as plots and heatmaps

2. **Preprocessing:**
   - Label encoding (e.g., 'B' mapped to 0, 'M' to 1)
   - Tensor conversion for model input

3. **Modeling:**
   - Neural network defined using PyTorch
   - Training and loss optimization
   - Evaluation with metrics like accuracy

4. **Results:**
   - Accuracy and loss plots
   - Model predictions on test data

## 🚀 How to Run

1. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn torch
```

2. Place your dataset in the same directory as the notebook and name it `data_day1.csv`.

3. Run all cells in the Jupyter notebook to:
   - Explore the data
   - Train the model
   - Evaluate its performance

## INFO

- The model is built from scratch using `torch.nn.Module`, showcasing core deep learning concepts.
- Useful for educational purposes or as a template for binary classification tasks.

## 📌 Author

Created during the **GEN AI DAY** workshop event for hands-on experience with data modeling and neural networks.

---

