
# 🩺 Diabetes Prediction using Decision Tree Regressor

This project focuses on predicting **diabetes progression** using a **Decision Tree Regression Model**. The goal is to use clinical data to estimate the disease progression metric for diabetes patients. The model is built and evaluated using **Scikit-learn** tools, and the entire project was developed using **Google Colab**.

## 🔍 Project Overview

- **Objective**: Predict the diabetes progression index using features like BMI, blood pressure, and age.
- **Dataset**: Inbuilt diabetes dataset from `sklearn.datasets`.
- **Model Used**: Decision Tree Regressor
- **Platform**: Google Colab

## 🧠 Key Concepts

- Supervised Machine Learning (Regression)
- Decision Trees
- Model Evaluation using RMSE, MAE, R² Score
- Data Visualization

## 📁 Project Structure

```
📦 Diabetes_Prediction_Decision_Tree
 ┣ 📜 Diabetes_Prediction_using_Decision_Tree_Regressor.ipynb
 ┗ 📄 README.md
```

## ⚙️ Technologies Used

- Python
- Jupyter / Google Colab
- NumPy
- Pandas
- Scikit-learn
- Matplotlib & Seaborn

## 📊 Dataset Details

- **Source**: `sklearn.datasets.load_diabetes()`
- **Target Variable**: A quantitative measure of disease progression one year after baseline.
- **Features**: Age, sex, BMI, average blood pressure, and six blood serum measurements.

## 🧪 Model Pipeline

1. **Load Dataset**
2. **Exploratory Data Analysis**
3. **Feature-Target Splitting**
4. **Train-Test Split**
5. **Model Training using DecisionTreeRegressor**
6. **Model Evaluation**:  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
   - Root Mean Squared Error (RMSE)  
   - R² Score

## ✅ Results

- The model performs reasonably well on the test set.
- Decision Trees tend to overfit, so results may vary depending on depth and train-test splits.

## 🚀 How to Run

1. Open in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

2. Upload and run the notebook:  
   `Diabetes_Prediction_using_Decision_Tree_Regressor.ipynb`

3. Make sure all dependencies are installed (most are available by default in Colab).

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Diabetes Dataset – sklearn.datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

## 🙌 Acknowledgements

- Developed on **Google Colab**
- Dataset courtesy of **Scikit-learn**
- Inspired by regression model tutorials in ML learning tracks
