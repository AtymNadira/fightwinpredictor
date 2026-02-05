# FightWinPredictor

FightWinPredictor is a machine learning project that predicts the outcome of professional MMA fights using athletes’ physical and demographic attributes. The system is trained on real UFC fight data and deployed as an interactive web application.

## Overview

The project explores how physical parameters such as height, weight, age, and fighting style influence match outcomes. Multiple classical machine learning models are trained and evaluated, with the best-performing model deployed via a Streamlit interface for real-time prediction.

## Features

- Data preprocessing and feature engineering on real UFC fight datasets  
- Training and evaluation of multiple ML models  
- Outcome prediction based on fighter physical attributes  
- Interactive web interface built with Streamlit  
- Clear visualization of results and model behavior  

## Machine Learning Pipeline

1. Data loading and cleaning  
2. Feature selection and encoding  
3. Model training (Logistic Regression, Decision Tree, Random Forest)  
4. Model evaluation and comparison  
5. Deployment of the best-performing model  

## Tech Stack

- Python  
- pandas  
- NumPy  
- scikit-learn  
- Streamlit  
- matplotlib
- seaborn 

## Installation

Clone the repository:
```bash

git clone https://github.com/your-username/fightwinpredictor.git
cd fightwinpredictor
streamlit run app.py

