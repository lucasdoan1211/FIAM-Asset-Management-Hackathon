# FIAM Asset Management Hackathon

## Overview
This repository contains the work developed for the **FIAM Asset Management Hackathon** held in **Montreal, Canada (Sept 2024 – Oct 2024)**. The project focuses on leveraging deep learning and advanced machine learning techniques to analyze large-scale financial data and enhance predictive accuracy for investment strategies.

## Project Goals
- **Develop a deep learning-based financial model** by performing extensive **data preprocessing, feature selection, and data analysis**.
- **Identify top 25% highly correlated features** from big financial data spanning **2000-2023**.
- **Improve predictive accuracy** through **hyperparameter tuning** and optimization.
- **Develop mixed buy-sell investment strategies** for **100 stocks**, enhancing fund performance evaluation.

## Key Features
- **Data Preprocessing**: Cleaning, normalization, and feature selection from large financial datasets.
- **Feature Engineering**: Identifying and extracting relevant financial indicators.
- **Machine Learning Models**: Leveraging **Deep Learning** and **advanced ML models** to optimize predictions.
- **Hyperparameter Tuning**: Fine-tuning models to maximize predictive performance.
- **Investment Strategy Simulation**: Applying **mixed buy-sell strategies** on selected stocks.

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch)
- **Deep Learning** (LSTM, RNN, Transformer-based models)
- **Machine Learning** (Random Forest, XGBoost, SVM, etc.)
- **Data Visualization** (Matplotlib, Seaborn)
- **Financial Data APIs** (Yahoo Finance, Alpha Vantage)

## Data and Methodology

### Data Preprocessing
The data preprocessing step ensures data integrity and usability by cleaning and selecting relevant factors and stocks based on missing values and zero values. Factors with fewer than **30% missing values** and **less than 20% zero values** are retained. Stocks are selected based on the number of available months, keeping those with the most available data and removing stocks with all missing values for any factor. Missing values in smaller gaps are filled using **mean or median imputation**, though this method might not fully capture temporal dynamics. Additionally, a **ranking and normalization process** is applied, where each factor is ranked based on its values to enable comparison between stocks. Normalization ensures that factors contribute equally to the ranking process, preventing scale bias.

### Feature Selection
Using **Robust Scaler** for feature scaling reduces the impact of outliers, which are common in financial data, ensuring that the model's performance is not skewed by extreme values. **Recursive Feature Elimination (RFE)** helps automatically identify and retain the most important features, enhancing interpretability and reducing overfitting. The use of **500 estimators and subsampling** helps generalize better on unseen data by preventing overfitting and speeding up training. **XGBoost's flexibility** with parameters like learning rate and tree sampling provides precise control over model performance, making it highly adaptable to various financial prediction tasks. While a rule of thumb suggests keeping **30-45 features** out of **145**, several tests identified **50 features** as optimal for maximizing the **R-squared** value.

### Predictive Model
**XGBoost** is an efficient **gradient boosting algorithm** that handles **non-linear relationships** in large, complex datasets. It uses **regularization and parallel processing** to reduce overfitting and computational cost. Hyperparameter tuning via **GridSearchCV** optimizes model performance by testing various parameter combinations, enhancing predictive accuracy. **Robust Scaler** is applied to reduce the impact of outliers, ensuring stability in complex financial data. The model employs **time series cross-validation and a time window approach**, ensuring realistic training and validation over time. These techniques strengthen **XGBoost's ability to handle feature interactions** and imbalances in data, improving its ability to capture complex patterns in financial datasets. Hyperparameter values are selected based on multiple tests to balance precision with moderate computational cost and training duration.

## LYTA Strategy Analytics: Portfolio Performance

### LYTA Portfolio vs. S&P 500 (2010-2024)
| Metric              | LYTA Portfolio | S&P 500 |
|--------------------|---------------|---------|
| **Average Annual Return** | **36.54%** | 13.42% |
| **Standard Deviation** | 0.1313 | 0.157 |
| **Alpha (CAPM)** | 0.0286 | 0 |
| **Sharpe Ratio** | **2.47** | Below 0.90 |
| **Information Ratio** | **2.61** | N/A |
| **Max Drawdown** | -23.02% | -18.11% |
| **Max 1-Month Loss** | -16.52% | -9.18% |
| **Turnover (Long)** | 35.09% | Below 5% |
| **Turnover (Short)** | 49.80% | N/A |

### Performance Summary
- The **LYTA Portfolio** achieves an **average return of 36.54%**, significantly outperforming the **S&P 500’s 13.42%**.
- Despite a **higher max drawdown (-23.02%)** than the S&P 500 (-18.11%), the **Sharpe ratio (2.47)** indicates strong **risk-adjusted returns**.
- The **Information Ratio (2.61)** highlights the model’s superior ability to generate excess returns relative to market benchmarks.

## Strategy Review

### Key Insights
- The model’s focus on **fundamental signals** like **market equity, price-to-high, and volatility** allowed it to capture **both upside potential and downside risk effectively**.
- **Alternative data and machine learning techniques** contributed to identifying patterns that traditional models might overlook.

### Profitable Stocks & Market Influence
- **Top-performing positions** included **First Solar and Hewlett-Packard (HP)**, which showed resilience during economic downturns.
- **Long positions** in companies with consistent revenue growth and low volatility provided stability, while **shorting overvalued stocks** capitalized on market corrections.
- The portfolio benefitted from **post-2008 financial crisis recovery**, the **stimulus-driven bull market (2010-2023)**, and **COVID-19 recovery opportunities** in **technology and consumer staples sectors**.

### Potential Improvements
- **Enhance Risk Management**: Implement **dynamic hedging strategies** to minimize drawdowns.
- **Feature Engineering**: Incorporate **natural language processing (NLP)** to analyze **financial news sentiment** and **earnings reports** for improved predictive accuracy.

