# Loan Default Prediction Using Real World Data

## What Is a Default?
Default is the failure to repay a debt, including interest or principal, on a loan or security. A default can occur when a borrower is unable to make timely payments, misses payments, or avoids or stops making payments. Individuals, businesses, and even countries can default if they cannot keep up their debt obligations. Default risks are often calculated well in advance by creditors.

## Predicting Defaults
As mentioned above, loan default risks are calculated in advance by the loan providers.<br>

For this project, real world data from an actual bank is used which provides Auto Loans based in USA.<br>

This project aims to develop a robust model for predicting your loan default risks. The features used for prediction are considered to be available at the time of loan origination and thus do not leak any information from the future.

## Modeling
Three machine learning algorithms were trained on the data, in which Catboost had the best result compared to other models, giving an accuracy of 86%.
And on looking at the feature importance we see Vehicle Model Year, Interest Rate, Vehicle New/Used and Payment to Income Percentage have higher feature weights,
giving us the factors that influence the default risks.

Link: [https://loan-default-predictions.herokuapp.com/](https://loan-default-predictions.herokuapp.com/).
