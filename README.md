# EV-Energy-Efficiency-Analysis-and-Prediction
EV Energy Efficiency Analysis and Prediction

EV Energy Efficiency Analysis and Prediction
Project Overview

This project analyzes electric vehicle (EV) specifications to understand the factors that influence vehicle energy efficiency. Using data from the Government of Canada EV database, the project applies data science techniques including data cleaning, exploratory data analysis, machine learning modeling, and AI-assisted interpretation to identify patterns in EV performance and develop predictive models for energy efficiency.

Energy efficiency is measured using the metric kilometers per kilowatt-hour (km/kWh), which represents how far an electric vehicle can travel per unit of electricity.

Dataset

Source: Government of Canada Electric Vehicle Database

The dataset includes specifications for battery electric vehicles manufactured between 2012 and 2026.

Key variables include:

Model year

Vehicle manufacturer and model

Vehicle class

Motor power (kW)

Vehicle range (km)

Recharge time (hours)

Energy consumption (kWh per 100 km)

A new variable, Energy Efficiency (km/kWh), was created during the analysis.

Project Workflow

Data Cleaning and Preparation

Exploratory Data Analysis

EV Efficiency Trend Analysis

Machine Learning Model Development

Model Performance Evaluation

Feature Importance Analysis

AI-Assisted Interpretation

Machine Learning Models

Two regression models were used to predict EV energy efficiency:

Linear Regression (baseline model)

Random Forest Regression

Model Performance
Model	RMSE	R²
Linear Regression	0.607	0.52
Random Forest	0.305	0.88

The Random Forest model significantly outperformed the Linear Regression model, demonstrating strong predictive performance.

Key Findings

Motor power is the most important predictor of EV energy efficiency.

Higher performance EVs tend to have lower energy efficiency.

Vehicle range and battery characteristics strongly influence efficiency.

EV efficiency has improved steadily across model years due to technological advances.

Tools and Technologies

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Generative AI tools for analytical interpretation

Author

Joshua Oscar Tetteh
Master of Applied Economics
Saint Mary's University, Canada
