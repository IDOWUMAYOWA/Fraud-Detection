# Fraud Detection System

A machine learning-based fraud detection system that identifies fraudulent financial transactions using logistic regression. The project includes exploratory data analysis, model training, and a user-friendly Streamlit web application for real-time fraud prediction.

## 📊 Project Overview

This project analyzes a dataset of over 6.3 million financial transactions to build a fraud detection model. The system focuses on identifying fraudulent TRANSFER and CASH_OUT transactions, which account for the majority of fraud cases in the dataset.

### Key Findings
- **Dataset Size**: 6,362,620 transactions
- **Fraud Rate**: 0.13% (highly imbalanced dataset)
- **Primary Fraud Types**: TRANSFER (0.77% fraud rate) and CASH_OUT (0.18% fraud rate)
- **Model Accuracy**: 94.67%
- **Fraud Detection Rate**: 94% recall on fraudulent transactions

## 🎯 Features

The model uses the following transaction features for prediction:

- **Transaction Type**: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
- **Amount**: Transaction amount
- **Old Balance (Sender)**: Sender's balance before transaction
- **New Balance (Sender)**: Sender's balance after transaction
- **Old Balance (Receiver)**: Receiver's balance before transaction
- **New Balance (Receiver)**: Receiver's balance after transaction

## 🏗️ Project Structure

```
fraud-detection/
├── fraud-detection.pdf          # Jupyter notebook with EDA and model training
├── app.py                        # Streamlit web application
├── fraud_detection_model.pkl     # Trained model (generated after training)
├── AIML Dataset.csv             # Dataset (not included in repo)
└── README.md                     # Project documentation
```

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

## 🚀 Usage

### 1. Model Training

Run the Jupyter notebook to train the model:

1. Open `fraud-detection.ipynb`
2. Execute all cells to:
   - Load and explore the dataset
   - Perform feature engineering
   - Train the logistic regression model
   - Save the model as `fraud_detection_model.pkl`

### 2. Running the Web Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### 3. Making Predictions

1. Select the transaction type from the dropdown
2. Enter the transaction amount
3. Input the sender's old and new balances
4. Input the receiver's old and new balances
5. Click "Predict" to see if the transaction is fraudulent

## 📈 Model Details

### Algorithm
**Logistic Regression** with the following configuration:
- Class weight: balanced (to handle class imbalance)
- Max iterations: 1000
- Preprocessing: StandardScaler for numerical features, OneHotEncoder for categorical features

### Data Preprocessing

1. **Feature Selection**: Removed non-predictive columns (step, nameOrig, nameDest, isFlaggedFraud)
2. **Feature Engineering**: Created BalanceOriginal and BalanceDestination features
3. **Encoding**: One-hot encoding for transaction type
4. **Scaling**: StandardScaler for numerical features
5. **Train-Test Split**: 70-30 split with stratification

### Model Performance

| Metric | Legitimate Transactions | Fraudulent Transactions |
|--------|------------------------|-------------------------|
| Precision | 1.00 | 0.02 |
| Recall | 0.95 | 0.94 |
| F1-Score | 0.97 | 0.04 |
| Support | 1,906,322 | 2,464 |

**Overall Accuracy**: 94.67%

**Confusion Matrix**:
- True Negatives: 1,804,823
- False Positives: 101,499
- False Negatives: 151
- True Positives: 2,313

### Key Insights

- The model successfully identifies 94% of fraudulent transactions (high recall)
- Low precision (2%) indicates many false positives, which is acceptable in fraud detection to minimize missed fraud cases
- The high false positive rate is a trade-off for catching more fraud cases
- Further tuning with ensemble methods or deep learning could improve precision

## 📊 Data Analysis Highlights

### Transaction Distribution
- **CASH_OUT**: 2,237,500 transactions (most common)
- **PAYMENT**: 2,151,495 transactions
- **CASH_IN**: 1,399,284 transactions
- **TRANSFER**: 532,909 transactions
- **DEBIT**: 41,432 transactions (least common)

### Fraud Patterns
- Fraud only occurs in **TRANSFER** and **CASH_OUT** transactions
- No fraud detected in PAYMENT, CASH_IN, or DEBIT transactions
- 1,188,074 transactions had suspicious patterns (zero balance after transfer)

### Correlations
- Strong correlation between old and new balances for destination accounts (0.98)
- Weak correlation between transaction amount and fraud (0.08)

## 🛠️ Future Improvements

1. **Model Enhancement**:
   - Try ensemble methods (Random Forest, XGBoost)
   - Experiment with deep learning approaches
   - Implement SMOTE or other resampling techniques

2. **Feature Engineering**:
   - Add time-based features
   - Include transaction velocity metrics
   - Create user behavior profiles

3. **Application Features**:
   - Add batch prediction capability
   - Include confidence scores
   - Implement transaction history tracking
   - Add data visualization dashboard

4. **Production Readiness**:
   - API development for integration
   - Model monitoring and retraining pipeline
   - A/B testing framework

## ⚠️ Limitations

- The model has a high false positive rate (2% precision for fraud class)
- Dataset is highly imbalanced, which affects model performance
- Model assumes feature distributions remain consistent over time
- No temporal features included in the current version

## 📝 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback about this project, please open an issue in the repository.

---

**Note**: Make sure to download the dataset and place it in the project directory before running the training notebook. The model file (`fraud_detection_model.pkl`) must be generated before running the Streamlit application.
