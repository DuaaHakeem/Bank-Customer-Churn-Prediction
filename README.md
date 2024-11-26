# Bank Customer Churn Prediction
This study focuses on building a predictive model to classify customers as churned or retained. It involves data preparation, feature engineering, model training, and evaluation. Key performance metrics, such as accuracy, precision, and recall, are used to assess the model's effectiveness. The ultimate goal is to provide actionable insights to enhance customer retention strategies while ensuring a seamless customer experience.

# Data Structure Overview
The data used in this study is 'bank customer churn prediction' from kaggle.
It contains customer information from a bank, including key attributes that may influence churn behavior. It is designed to capture various aspects of customer profiles, interactions, and account details. Below is an outline of the dataset's structure:

**Customer Demographics:**

**Customer ID:** A unique identifier for each customer. 

**Surname:** The customer's surname or last name.

**Age:** The customer's age.

**Gender:** The customer's gender (Male or Female).

**Geography:** The country where the customer resides (France, Spain or Germany).

**Financial Attributes:**

**Balance:** The customer's account balance.

**EstimatedSalary:** The estimated salary of the customer.

**Credit Score:** A numerical value representing the customer's credit score.

**NumOfProducts:** The number of bank products the customer uses (e.g., savings account, credit card).

**Customer Behavior:**

**HasCrCard:** Whether the customer has a credit card (1 = yes, 0 = no).

**IsActiveMember:** Whether the customer is an active member (1 = yes, 0 = no).

**Tenure:** The number of years the customer has been with the bank.

**Target Variable:**

**Exited:** Whether the customer has churned (1 = yes, 0 = no).

This dataset provides a comprehensive view of the factors that may contribute to churn. With its mix of numerical and categorical features, it allows for exploratory data analysis, feature engineering, and the application of machine learning algorithms to predict customer churn.
During the data preparation phase, missing values, outliers, and any imbalances in the target variable are addressed to ensure robust model training. Exploratory analysis is conducted to identify trends and correlations, helping uncover critical factors driving customer attrition.
By analyzing this dataset, the study aims to generate actionable insights to assist the bank in developing data-driven strategies to retain valuable customers.

# Exploratory Data Analysis (EDA)
 EDA guide subsequent stages of model building, including feature selection, data preprocessing, and model choice, as it helps discover validate assumptions, identify outliers, and understand variable relationships, which are crucial for improving the performance of the classification model. This exploratory phase is essential to enhances model accuracy and mitigates risks of overfitting or bias by ensuring data quality and relevance. Below is a summary of the findings:

**1. Data Quality and Cleaning**
    
**Missing Values:** There is one missing value in each of the following columns: Geography, Age, HasCrCard, and IsActiveMember. Since the dataset only has four rows with missing values, I can either drop these rows without significantly impacting the dataset’s overall size or impute the missing values. Imputing these values using the mean for numeric variables and the mode for categorical variables would also maintain data quality, as the small amount of missing data has minimal impact.

**Duplicate Records:** Two duplicate records were found.

**2. Distribution of Features**

**Age:** The Age column shows evidence of skewness, which may violate the assumptions of a logistic regression model. A data transformation might be necessary to address this issue and ensure the model's assumptions are met.

**Gender:** The dataset contains a balanced proportion of male and female customers. However, female customers exhibited a slightly higher churn rate.

By observing the distrbution of numeric features across categories in target variable, majority of the features may not provide significant information about the target.

**3. Financial Behavior**

**Balance:** There is a high frequency of zero values in the Balance column, which may stands out as unusual. This was already noted during the review of the first 10 rows of data. Further investigation is required to understand why so many customers have a zero balance and how this correlates with other features.

**Estimated Salary:** Salary distribution appeared uniform across the dataset, with no clear correlation to churn.

**Number of Products:** Customers with one product had a higher tendency to churn, while those with multiple products were more likely to remain.

**4. Customer Activity and Engagement**

**IsActiveMember:** Active members showed a slightly lower churn rate compared to inactive members.

**Tenure:** Customers with shorter tenures (e.g., under 3 years) were more likely to churn compared to those with longer tenures.

**5. Correlation Analysis**
A weak correlation was observed between most features and churn, emphasizing the need for advanced modeling techniques to capture non-linear relationships.
No evidence of multicollinearity was found among the numerical features, ensuring that all variables could be retained for modeling.

**6. Class Imbalance**
The target variable (Exited) was imbalanced, with a larger proportion of retained customers compared to churned ones. This imbalance will be addressed during model training using resampling techniques (e.g., SMOTE).

# Feature Engineering
In this section, I apply feature engineering techniques to create new features based on existing columns in the dataset. These engineered features will help to capture more meaningful relationships in the data and improve model performance.

# Model Comparison
In the phase, I trained machine learning models, including Logistic Regression, Decision Tree and Random Forest to identify the most effective model. The model that demonstrates the best initial performance will undergo hyperparameter tuning via GridSearchCV to further enhance its efficacy. I trained the models before and after data resampling to see how imbalance data can effect the model performance.

**1. Logistic Regression:**

 I used it as a baseline model for predicting customer churn.
 
 **Model Performance:**
 
 **Non-linear Relationships:** As observed during exploratory data analysis (EDA), many features exhibit non-linear relationships with the target variable (e.g., churn rate varying across tenure, balance, and activity levels). Logistic Regression, being a linear model, struggles to capture these complex patterns.
 
**Class Imbalance:**
The dataset is imbalanced, with a higher proportion of retained customers (class 0) compared to churned customers (class 1).
As a result, the model is biased toward predicting the majority class, achieving high precision and recall for retained customers but failing to generalize for the minority class (churned customers). This is evident in the low recall (22%) and F1-score (31%) for the churned class.

**2. Decision Tree:**
The Decision Tree model demonstrates improved performance over Logistic Regression by capturing non-linear relationships in the data. However, its effectiveness is still impacted by the class imbalance, leading to suboptimal recall for the minority class.

**3. Random Forest:**
The Random Forest model outperforms both Logistic Regression and Decision Tree models in overall accuracy, precision, and F1-score. It demonstrates its capability to handle non-linear relationships effectively and performs better in predicting the minority class. However, class imbalance remains a challenge, especially for recall.

# Train models with Resampling:
**Decision Tree:** The Resampled Decision Tree model demonstrates significant improvements in handling class imbalance, achieving balanced performance for both classes while maintaining high overall accuracy. It outperforms the original Decision Tree model in terms of fairness and effectiveness for the churned class.

**Random Forest:** The Resampled Random Forest model stands out as the best-performing model overall, with high accuracy and balanced precision and recall for both classes. Its ability to capture non-linear relationships and mitigate class imbalance makes it highly effective for bank churn prediction.
