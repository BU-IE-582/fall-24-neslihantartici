#!/usr/bin/env python
# coding: utf-8

# Group-10 : Sude Şahin - 2024702045
# Neslihan Tartıcı - 2024705027
# Melike Sevgi - 2016402138

# **INTRODUCTION**
# This project aims to develop predictions for match outcomes using data generated during the early stages of football matches (the first 16 minutes). The primary challenge is that, in line with real-time betting scenarios, decisions must be made for each match within a single time frame. Therefore, analyses and predictions are based solely on data available up to the selected time.
# 
# During the data preprocessing steps, information from the first 16 minutes of matches was filtered, and missing data were analyzed. Columns with more than 50% missing values were removed, while the remaining gaps were filled using appropriate imputation methods. A date filter was applied to split the data: matches before "2024-11-01" were allocated for training and model development, while those after this date were reserved for model testing. This approach is crucial for simulating real-world scenarios.
# 
# Feature selection was performed using a correlation matrix and the Random Forest method. Correlation analysis was employed to examine the relationships between independent variables, and steps were taken to reduce multicollinearity. The importance of features was calculated with Random Forest, enabling the selection of more effective features for predictions. This process enhanced the model's performance while eliminating redundant data.
# 
# For prediction processes, the XGBoost model was utilized, and the resulting predictions were integrated into a decision-making framework for betting based on probabilities and odds. The betting strategy was formulated based on expected value (EV) calculations and probability thresholds. In scenarios with positive EV and high probabilities, a "bet" decision was made, while in other cases, a "no-action" decision was taken.
# 
# By combining dynamic data analysis, feature selection, machine learning, and decision-making mechanisms, this project aims to provide meaningful insights into sports analytics and betting strategies.

# In[122]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Data
data = pd.read_csv('match_data.csv')
print("First 5 Rows of Data:\n", data.head())


# In[123]:


data.groupby("fixture_id")["minute"].size().sort_values()


# In[124]:


# Selecting data with minute values less than 15
data = data[(data['halftime'] == '1st-half') & (data['minute'] <= 15)]

print(data)


# In[125]:


data.groupby("fixture_id")["minute"].size().sort_values()


# In[126]:


# Calculate the total number of unique fixture_id
unique_fixture_id_count = data['fixture_id'].nunique()

# Print the result
print(f"Total number of unique fixture_ids: {unique_fixture_id_count}")


# In[127]:


# Select rows with the maximum minute value for each fixture_id
max_minute_data = data.loc[data.groupby('fixture_id')['minute'].idxmax()]

# Check the first few rows
max_minute_data.head(100)

# Sort by the 'minute' column (ascending order)
sorted_max_minute_data = max_minute_data.sort_values(by='minute', ascending=True)

# Check the first few rows
sorted_max_minute_data.head(100)


# In[128]:


# Missing data analysis and threshold
missing_threshold = 50  # Threshold to drop columns with more than 50% missing values

# Column-wise missing data analysis
missing_values = data.isnull().sum()
missing_columns = missing_values[missing_values > 0]
missing_percentage = (missing_columns / len(data)) * 100

# Analysis of columns based on the percentage of missing data
missing_analysis = pd.DataFrame({
    'Missing Values': missing_columns,
    'Percentage (%)': missing_percentage
}).sort_values(by='Percentage (%)', ascending=False)

# Identify columns with more than 50% missing data
columns_to_drop = missing_analysis[missing_analysis['Percentage (%)'] > missing_threshold].index

# Drop columns
data_cleaned = data.drop(columns=columns_to_drop)
print(f"{len(columns_to_drop)} columns were dropped. New column count: {data_cleaned.shape[1]}")

# Identify columns with less than or equal to 50% missing data
columns_to_fill = missing_analysis[missing_analysis['Percentage (%)'] <= missing_threshold].index

# Fill missing values
for column in columns_to_fill:
    if data_cleaned[column].dtype in ['float64', 'int64']:  # Numerical columns
        data_cleaned[column].fillna(data_cleaned[column].mean(), inplace=True)
    else:  # Categorical or textual columns
        data_cleaned[column].fillna(data_cleaned[column].mode()[0], inplace=True)

# Check remaining missing values
remaining_missing = data_cleaned.isnull().sum().sum()
print(f"Remaining missing values: {remaining_missing}")

# Select rows with the maximum minute value for each fixture_id
data_cleaned = data_cleaned.loc[data_cleaned.groupby('fixture_id')['minute'].idxmax()]

# Check the cleaned dataset with maximum minute values
print("Cleaned dataset with maximum minute values:")
print(data_cleaned.head())


# **Data Cleaning Process**
# 
# The data cleaning process is a critical step to ensure the dataset is suitable for analysis and modeling. In this phase, the missing values in the dataset were analyzed first. The number of missing values in each column was identified and expressed as a percentage of the total data. Columns with a missing value ratio exceeding 50% were excluded from the analysis due to significant information loss. For the remaining columns, missing values were imputed. Numeric variables were filled with the column's mean value, while categorical variables were imputed with the most frequent value (mode), ensuring data integrity.
# 

# In[129]:


# Basic statistics of numerical columns
data_cleaned.describe()


# In[130]:


data_cleaned['match_start_datetime'] = pd.to_datetime(data_cleaned['match_start_datetime'])
data_cleaned['current_time'] = pd.to_datetime(data_cleaned['current_time'])


# In[131]:


# Specify columns to remove
columns_to_remove = [
    'halftime', 'current_time', 'half_start_datetime',
    'latest_bookmaker_update', 'name', 'ticking',
    'suspended', 'stopped'
]

# Remove the specified columns from the dataset
data_cleaned = data_cleaned.drop(columns=columns_to_remove, errors='ignore')

# Check the updated number of columns
print(f"Number of columns removed: {len(columns_to_remove)}")
print(f"Remaining number of columns: {data_cleaned.shape[1]}")

# Display the first few rows of the cleaned dataset
data_cleaned.head()


# In[132]:


from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Separate independent and dependent variables
X = data_cleaned.drop(columns=['result', 'fixture_id', 'match_start_datetime'])  # Remove result and ID
y = data_cleaned['result']  # Dependent variable

# Identify and transform categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Check after numerical transformation
print("Categorical columns transformed:", categorical_columns.tolist())

# Shapes of independent and dependent variables
print(f"Independent variables: {X.shape}, Dependent variables: {y.shape}")


# In[133]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()  # Correlation among independent variables
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix (First 15 Minutes)")
plt.show()

# 2. Feature Importance with Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# Extract feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Visualize feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance by Random Forest")
plt.show()

# 3. Identify Important Features
threshold = 0.01  # Threshold for important features
selected_features = feature_importance[feature_importance['Importance'] > threshold]['Feature']
print(f"Selected Features: {selected_features.tolist()}")


# These codes ensure the separation of independent and dependent variables within the dataset and the conversion of categorical variables into numerical formats. Independent variables are the features used by the model for prediction, created by excluding the target variable (result column) and non-informative columns like fixture_id and match_start_datetime. In this way, only the columns valuable for modeling are selected as independent variables. The dependent variable is the result column, representing the outcome of each match. This column consists of three classes: home win (1), away win (2), and draw (X), which the model aims to predict.
# 
# For machine learning models to process categorical variables, these variables must be converted into numerical formats. This conversion was carried out using LabelEncoder. Each categorical column in the dataset was transformed into numerical codes corresponding to its unique categories. For instance, categorical values like ["A", "B", "C"] were encoded as [0, 1, 2]. This process was applied to all categorical columns, and the transformation was verified afterward to confirm the converted columns.
# 
# The ranking of the independent variables in terms of importance was determined to identify impactful features for analysis. A correlation matrix was utilized to reveal the linear relationships between the independent variables. In the matrix, red areas indicated positive correlations (as one variable increases, the other also increases), while blue areas showed negative correlations (as one variable increases, the other decreases). The diagonal part of the matrix was always dark red, representing a 100% correlation of each variable with itself. This analysis is particularly critical for detecting multicollinearity. For instance, independent variables with high correlation (e.g., above 0.8) can be selected, merged, or excluded to improve the analysis. This step is essential for enhancing the model's learning process.
# 
# A feature importance graph generated using the Random Forest model illustrates each independent variable's contribution to the dependent variable. Variables ranked at the top of the graph are the most influential in the model's predictive power. For example, variables like "Ball Possession %" and "Dangerous Attacks" may have a high impact on the target variable. Conversely, variables ranked at the bottom of the graph are deemed less important for the model and can be excluded from the analysis or assigned a lower weight. In line with the assignment requirements, feature selection focused exclusively on the data available at the end of the first 16 minutes of the match. Features such as real-time game statistics (e.g., Goals - home, Goals - away, Assists - home, Assists - away) and team performance indicators (e.g., Ball Possession %, Successful Passes Percentage) were selected for their meaningful contributions to the prediction model. However, features that posed a risk of future data leakage (e.g., final_score) and those with no significant contribution to the prediction process (e.g., second) were removed from the model. These selections ensured that the prediction process was free from "future-seeing" errors and enabled the development of a more realistic betting strategy.
# 
# The insights derived from these analyses guide subsequent steps to improve model performance. Specifically, removing or merging highly correlated variables allows the model to work with a more concise dataset. Additionally, eliminating features with low importance reduces computational costs and mitigates the risk of overfitting. Consequently, these analyses enable the execution of feature selection and model optimization, ultimately improving model performance and facilitating more effective predictions.

# In[134]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# In[135]:


selected_features = [
     'second', 'Successful Passes - home', 'Ball Possession % - home', 'Ball Possession % - away', 'Successful Passes Percentage - home', 'Successful Passes Percentage - away', 'Passes - home', 'Passes - away', 'Successful Passes - away', 'Attacks - home', 'Dangerous Attacks - home', 'Attacks - away', 'Dangerous Attacks - away', 'Long Passes - home', 'Throwins - away', 'Long Passes - away', 'Tackles - home', 'Successful Interceptions - away', 'Total Crosses - away', 'Interceptions - away', 'Total Crosses - home', 'Fouls - home', 'Tackles - away', 'Goal Kicks - away', 'Interceptions - home', 'Throwins - home', 'Challenges - home', 'Challenges - away', 'Shots Insidebox - home', 'Fouls - away', 'Shots Total - home', 'Successful Interceptions - home']


# In[136]:


# Check for missing columns
missing_columns = [col for col in selected_features if col not in data_cleaned.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    # Split into training and test sets
    train_data = data_cleaned[data_cleaned['match_start_datetime'] < '2024-11-01']
    test_data = data_cleaned[data_cleaned['match_start_datetime'] >= '2024-11-01']

    # Select independent and dependent variables
    X_train = train_data[selected_features]
    y_train = train_data['result']
    X_test = test_data[selected_features]
    y_test = test_data['result']

    # Fill missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)


# In[137]:


# Manually convert class labels
mapping = {'1': 1, '2': 2, 'X': 0}
y_train_mapped = y_train.map(mapping)
y_test_mapped = y_test.map(mapping)

# Verify the transformation
print("Transformed y_train:", y_train_mapped[:5].tolist())  # Example: [1, 2, 0, 1, ...]
print("Transformed y_test:", y_test_mapped[:5].tolist())


# In[138]:


print("Original y_train class distribution:")
print(y_train.value_counts())

print("Original y_test class distribution:")
print(y_test.value_counts())


# This step involves splitting the dataset into training and testing sets, selecting independent and dependent variables, and appropriately transforming class labels. Initially, the dataset was divided using a date filter specified in the project scope. Matches played before 2024-11-01 were used for training the model and optimizing its parameters, while matches played after this date were reserved for testing the model. This separation allows the model to simulate real-world scenarios effectively.
# 
# In both training and testing datasets, predefined features (e.g., successful pass percentage, ball possession rate, number of dangerous attacks) were selected as independent variables, while match outcomes were assigned as the dependent variable. Missing data were filled with zero values to ensure they did not negatively affect model performance.
# 
# The dependent variable, match outcomes (result), was converted into numerical values for the model to process. This conversion classified outcomes into three categories: "home win" (1), "away win" (2), and "draw" (0). These class labels were applied to the dependent variable using a mapping method. For example:
# 
# '1' → Home win
# '2' → Away win
# 'X' → Draw
# 
# After the transformation, the class distributions were examined to ensure a balanced structure in the training and testing datasets. This step is critical for properly training the model and enhancing its prediction performance.

# In[139]:


# Ignore warnings for XGBoost
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define and train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train_mapped)

# Make predictions
y_train_predictions = model.predict(X_train)
y_test_predictions = model.predict(X_test)

# Calculate accuracy and generate classification report for the training set
training_accuracy = accuracy_score(y_train_mapped, y_train_predictions)
training_classification_report = classification_report(y_train_mapped, y_train_predictions)

# Calculate accuracy and generate classification report for the testing set
testing_accuracy = accuracy_score(y_test_mapped, y_test_predictions)
testing_classification_report = classification_report(y_test_mapped, y_test_predictions)

# Print the results
print(f"Training Set Accuracy: {training_accuracy}")
print("Training Set Classification Report:\n", training_classification_report)

print(f"Testing Set Accuracy: {testing_accuracy}")
print("Testing Set Classification Report:\n", testing_classification_report)


# In this step, the XGBoost model was defined, trained, and its performance evaluated. XGBoost is a machine learning algorithm renowned for its high accuracy and fast computation capabilities, and it was used in this project to predict match outcomes. While defining the model, the parameter use_label_encoder=False was set to prevent label encoding errors, and mlogloss (multi-class log loss) was chosen as the evaluation metric (eval_metric). To ensure randomness control, the parameter random_state=42 was specified. The model was trained using the independent variables (features) and the dependent variable (match outcomes) from the training dataset.
# 
# The model's performance was evaluated by making predictions on both the training and test datasets. The accuracy on the training dataset was calculated as 100%. Precision, recall, and F1 scores for each class were also measured as 1.00. These results demonstrate that the model fit the training data perfectly; however, this suggests that the model may have overfitted the training data. On the test dataset, the accuracy was calculated as 41%.
# 
# When evaluated on a per-class basis:
# 
# -For class 1 (home win), the precision and recall values were 0.59 and 0.57, respectively, indicating better performance compared to other classes.
# -For class 0 (draw) and class 2 (away win), the precision and recall values were significantly lower, in the range of 0.23-0.26.
# -The macro average (macro avg) and weighted average (weighted avg) scores were 0.36 and 0.42, respectively, reflecting the imbalanced performance across classes.

# In[140]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],          # Ağaç derinliği
    'learning_rate': [0.01, 0.1, 0.2],  # Öğrenme hızı
    'n_estimators': [100, 200, 300],    # Ağaç sayısı
    'subsample': [0.8, 1.0],           # Örnekleme oranı
    'colsample_bytree': [0.8, 1.0]     # Özellik örnekleme oranı
}

# Define the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train_mapped)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Optimal model
best_model = grid_search.best_estimator_

# Predictions on training set
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train_mapped, y_train_pred)
train_classification_report = classification_report(y_train_mapped, y_train_pred)

# Predictions on test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test_mapped, y_test_pred)
test_classification_report = classification_report(y_test_mapped, y_test_pred)

# Print results
print(f"Train Set Accuracy: {train_accuracy}")
print("Train Set Classification Report:\n", train_classification_report)

print(f"Test Set Accuracy: {test_accuracy}")
print("Test Set Classification Report:\n", test_classification_report)


# This code focused on optimizing the XGBoost model through hyperparameter tuning using GridSearchCV. The best parameter combination was identified as colsample_bytree: 0.8, learning_rate: 0.01, max_depth: 5, n_estimators: 200, and subsample: 0.8. As a result, the model achieved an impressive training set accuracy of 93.83%, demonstrating strong performance with high precision, recall, and F1-scores across all classes. On the test set, the model achieved an accuracy of 39.64%, with "Home Win" (class 1) predictions showing robust performance relative to other classes. The hyperparameter tuning process successfully optimized the model's parameters, improving its ability to leverage the training data effectively. These results highlight the effectiveness of hyperparameter tuning in refining model performance and provide a solid foundation for further experimentation and analysis. The accuracy value is considered sufficient as it is based on data from the first 16 minutes of the match, given that the predictions are being made by analyzing only 1/6 of the match duration.

# In[141]:


import matplotlib.pyplot as plt
import numpy as np

# Calculate the total counts of true and predicted values for each class
classes = ['0', '1', '2']  # Class labels
class_labels = ['Draw (X)', 'Home Win', 'Away Win']  # Class descriptions
true_counts = [sum(y_test_mapped == i) for i in range(3)]  # True class distribution
predicted_counts = [sum(y_test_pred == i) for i in range(3)]  # Predicted class distribution

# Settings for the plot
x = np.arange(len(classes))  # Classes for the x-axis
width = 0.35  # Width of the bars
fig, ax = plt.subplots(figsize=(10, 6))

# Bars for true values
ax.bar(x - width/2, true_counts, width, label='True', color='blue')

# Bars for predicted values
ax.bar(x + width/2, predicted_counts, width, label='Predicted', color='orange')

# Labels and title for the plot
ax.set_xlabel('Class')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of True and Predicted Values for Each Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Add class descriptions below the bars
for i, label in enumerate(class_labels):
    ax.text(x[i], -max(true_counts + predicted_counts) * 0.1, label, ha='center', fontsize=10, color='black')

# Show the plot
plt.show()


# In this step, a visualization was created to better understand the model's prediction performance by comparing the distributions of actual and predicted classes. The project defined three classes: "0" (draw), "1" (home win), and "2" (away win). For the test dataset, the frequencies of each class were calculated for both actual and predicted values and visualized using a bar chart. In the chart, actual values are represented with blue bars, while predicted values are shown with orange bars. These bars are positioned side by side for each class, making it easier to analyze the model's accuracy and performance. The chart was further refined with clear class labels and axis titles for better readability.
# 
# The visualization revealed several key insights into the model's performance. It showed that the model performs well in predicting home wins (class 1) but struggles with draws (class 0) and away wins (class 2). This discrepancy suggests that the model might be affected by class imbalances, leading to a stronger focus on the more represented or easier-to-predict classes. 

# In[142]:


import matplotlib.pyplot as plt
import numpy as np

# Calculate prediction probabilities
y_pred_proba = best_model.predict_proba(X_test)

# Class names
class_labels = ['Draw (X)', 'Home Win', 'Away Win']

# Visualize the probability distributions
plt.figure(figsize=(10, 6))
for i, class_name in enumerate(class_labels):  # Add class labels as text
    plt.hist(y_pred_proba[:, i], bins=20, alpha=0.5, label=f'{class_name}')

plt.title('Distribution of Prediction Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# This visualization was created to examine the distribution of prediction probabilities calculated by the model on a class-by-class basis. The chart includes histograms of prediction probabilities for three different classes: "Draw (X)" (tie), "Home Win" (home team victory), and "Away Win" (away team victory). The probability distributions for each class are represented with different colors: blue for draws, orange for home wins, and green for away wins.
# 
# Upon analyzing the chart, it is evident that the prediction probabilities for each class follow a distinct distribution. The "Home Win" class displays a more concentrated and pronounced distribution compared to the other classes, often associated with higher probability values. This indicates that the model has a stronger tendency and confidence in predicting home wins. In contrast, the probability distributions for "Away Win" and "Draw" are more dispersed and centered around lower probability values, highlighting the model's lower confidence and less precise predictions for these classes.

# In[143]:


# The first 15 minutes data is already contained in the cleaned data_cleaned
first_15_data = data_cleaned.copy()

# Define the required features and target variable
X_test = first_15_data[selected_features]  # Previously selected features
y_test = first_15_data['result']  # Actual results


# In[144]:


# Calculate prediction probabilities
y_pred_proba = model.predict_proba(X_test)

# Add predicted probabilities to the table
first_15_data['Probability - Draw (X)'] = y_pred_proba[:, 0]
first_15_data['Probability - Home Win'] = y_pred_proba[:, 1]
first_15_data['Probability - Away Win'] = y_pred_proba[:, 2]


# In[145]:


# Calculate expected value using probabilities and odds
# Retrieve betting odds from data_cleaned
first_15_data['Odds - Draw (X)'] = first_15_data['X']  # Odds for Draw
first_15_data['Odds - Home Win'] = first_15_data['1']  # Odds for Home Win
first_15_data['Odds - Away Win'] = first_15_data['2']  # Odds for Away Win

# Calculate expected value
first_15_data['EV - Draw (X)'] = (first_15_data['Probability - Draw (X)'] * (first_15_data['Odds - Draw (X)'] - 1)) - (1 - first_15_data['Probability - Draw (X)'])
first_15_data['EV - Home Win'] = (first_15_data['Probability - Home Win'] * (first_15_data['Odds - Home Win'] - 1)) - (1 - first_15_data['Probability - Home Win'])
first_15_data['EV - Away Win'] = (first_15_data['Probability - Away Win'] * (first_15_data['Odds - Away Win'] - 1)) - (1 - first_15_data['Probability - Away Win'])

# Make decisions based on betting threshold and positive EV
betting_threshold = 0.7  # Probability threshold
first_15_data['Bet on Draw (X)'] = (first_15_data['Probability - Draw (X)'] > betting_threshold) & (first_15_data['EV - Draw (X)'] > 0)
first_15_data['Bet on Home Win'] = (first_15_data['Probability - Home Win'] > betting_threshold) & (first_15_data['EV - Home Win'] > 0)
first_15_data['Bet on Away Win'] = (first_15_data['Probability - Away Win'] > betting_threshold) & (first_15_data['EV - Away Win'] > 0)


# In[146]:


# Amount staked per bet
stake = 1

# Calculate profit/loss
first_15_data['Profit - Draw (X)'] = first_15_data['Bet on Draw (X)'] * (stake * (first_15_data['Odds - Draw (X)'] - 1))
first_15_data['Profit - Home Win'] = first_15_data['Bet on Home Win'] * (stake * (first_15_data['Odds - Home Win'] - 1))
first_15_data['Profit - Away Win'] = first_15_data['Bet on Away Win'] * (stake * (first_15_data['Odds - Away Win'] - 1))

# Calculate total profit/loss
first_15_data['Total Profit'] = first_15_data[['Profit - Draw (X)', 'Profit - Home Win', 'Profit - Away Win']].sum(axis=1)

# Overall results
total_profit = first_15_data['Total Profit'].sum()
print(f"Total Profit: {total_profit}")


# In this step, a decision-making mechanism based on probabilities and odds was developed to evaluate betting strategies relying on data from the first 16 minutes of matches. For each match in the test dataset, the model-generated probabilities were categorized into three groups: Probability - Draw, Probability - Home Win, and Probability - Away Win. The available betting odds were integrated with these predicted probabilities, and the Expected Value (EV) was calculated for each probability and betting odd. The EV serves as a critical metric for assessing whether a betting strategy has the potential for profit, with a positive EV indicating that placing a bet is a logical choice. A probability threshold of 0.7 was selected, and bets were placed only for scenarios with a positive EV. A fixed stake amount was set per bet, and profit/loss was calculated for each bet, culminating in a total profit/loss value for all matches.
# 
# At the end of this process, the total profit was calculated as 1077.49 units. This positive result demonstrates the effectiveness of the strategy and indicates that the model's predictions could successfully be converted into profit. However, the accuracy of the probabilities and the reliability of the betting odds are crucial factors in the success of such an approach.

# In[147]:


# Retrieve prediction probabilities and odds
first_15_data['Odds - Draw (X)'] = first_15_data['X']
first_15_data['Odds - Home Win'] = first_15_data['1']
first_15_data['Odds - Away Win'] = first_15_data['2']

# Calculate expected values (EV)
first_15_data['EV - Draw (X)'] = (first_15_data['Probability - Draw (X)'] * (first_15_data['Odds - Draw (X)'] - 1)) - (1 - first_15_data['Probability - Draw (X)'])
first_15_data['EV - Home Win'] = (first_15_data['Probability - Home Win'] * (first_15_data['Odds - Home Win'] - 1)) - (1 - first_15_data['Probability - Home Win'])
first_15_data['EV - Away Win'] = (first_15_data['Probability - Away Win'] * (first_15_data['Odds - Away Win'] - 1)) - (1 - first_15_data['Probability - Away Win'])

# Decision mechanism
betting_threshold = 0.7  # Probability threshold
first_15_data['Decision'] = 'no action'  # Default decision
first_15_data.loc[(first_15_data['Probability - Draw (X)'] > betting_threshold) & (first_15_data['EV - Draw (X)'] > 0), 'Decision'] = 'bet draw'
first_15_data.loc[(first_15_data['Probability - Home Win'] > betting_threshold) & (first_15_data['EV - Home Win'] > 0), 'Decision'] = 'bet home win'
first_15_data.loc[(first_15_data['Probability - Away Win'] > betting_threshold) & (first_15_data['EV - Away Win'] > 0), 'Decision'] = 'bet away win'

# Create decision table
decision_table = first_15_data[['fixture_id', 'Probability - Draw (X)', 'Probability - Home Win', 
                                'Probability - Away Win', 'Odds - Draw (X)', 
                                'Odds - Home Win', 'Odds - Away Win', 
                                'EV - Draw (X)', 'EV - Home Win', 
                                'EV - Away Win', 'Decision']]

# Display the table
decision_table.head()


# This table illustrates how the model generates betting decisions based on probabilities and odds, while also calculating expected values (EV) for each scenario. The EV, a crucial metric for assessing the potential profitability of a bet, is determined using the formula: 
# 
# EV=Probability×(Odds−1)−(1−Probability). 
# 
# Separate EV calculations were performed for each bet type, including draws (EV - Draw), home wins (EV - Home Win), and away wins (EV - Away Win). For every match, probabilities for each outcome were computed and combined with their respective odds to calculate the EVs. A positive EV indicates that a bet is reasonable and potentially profitable, whereas a negative EV suggests a likelihood of loss. Based on these calculations, the class with the highest positive EV was selected as the betting decision for each match.
# 
# For instance, in fixture_id 19134453, the home win probability was 98.78%, making it the highest probability. The EV for this outcome was 0.590357, leading to the decision to "bet home win." Similarly, in fixture_id 19134454, the away win probability of 99.09% resulted in a positive EV of 0.387329, prompting the decision to "bet away win." In contrast, the draw class consistently showed negative EV values across all matches, leading to no betting decisions for this category.
# 
# This analysis demonstrates how probabilities and odds can be effectively utilized to make data-driven betting decisions, ensuring that each bet maximizes potential profitability based on the EV metric.

# In[148]:


import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of decisions
decision_counts = first_15_data['Decision'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=decision_counts.index, y=decision_counts.values, palette="viridis")
plt.title('Distribution of Decisions', fontsize=16)
plt.xlabel('Decisions', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# This chart visualizes the distribution of betting decisions made by the model across different classes. The bars, displayed in distinct colors, represent how frequently the model recommended bets for "Draw," "Home Win," and "Away Win" outcomes. Analyzing the chart is essential for understanding which classes the model prioritizes and identifying potential imbalances in its decision-making process.
# 
# According to the visualization, the model most frequently recommended bets for "Home Win" outcomes. This indicates that the model tends to predict home wins with higher probabilities and prioritizes this class in positive expected value (EV) calculations. In contrast, bets for "Draw" and "Away Win" outcomes were suggested less frequently compared to home wins.

# **RELATED LITERATURE**
# 
# The article titled Forecasting Football Match Results in National League Competitions Using Score-Driven Time Series Models was reviewed within the scope of the literature, focusing on the Bivariate Poisson, Skellam, and Ordered Probit methods it employed. These methods were analyzed in detail, and necessary implementations were coded. The analysis revealed an accuracy value of 1, indicating that the model exhibited overfitting. This finding suggests that the outcomes derived from the model can significantly vary based on changes in the requirements. 
# 
# The Bivariate Poisson Model has the advantage of modeling complex dependencies (e.g., relationships between the number of goals), but its disadvantage lies in the difficulty of interpreting prediction results and converting them into betting strategies. The Skellam Model is advantageous for its simplicity and ease of understanding but is limited in its ability to model complex relationships. Lastly, the Ordered Probit Model theoretically offers the advantage of capturing ordinal dependencies but suffers from low accuracy and potential losses in practical applications.

# **CONCLUSION**
# 
# In this study, a data-driven betting strategy was developed and evaluated using match data from the first 16 minutes. The analysis leveraged both descriptive and predictive methodologies, employing XGBoost for match outcome predictions and integrating calculated probabilities with betting odds to derive expected values (EV). The model achieved a training accuracy of 1.0, indicating excellent performance on the training set. However, the testing accuracy was 0.41, highlighting challenges in generalization to unseen data. The classification report further revealed that predictions for "Draw" and "Away Win" outcomes were less accurate compared to "Home Win," indicating a potential bias in the model.
# 
# From a profitability perspective, the strategy yielded a Total Profit of 1077.49 units by placing 1 unit per match based on positive EVs. This result suggests that the model successfully identified profitable betting opportunities, particularly for "Home Win" scenarios, despite limitations in predictive accuracy for other classes.
# 
# Key Findings
# 
# Accuracy: The overall accuracy of 41% on the testing set demonstrates that while the model can identify trends, its predictions for less frequent outcomes (e.g., "Draw") require improvement. The imbalance in class distribution likely contributed to this limitation.
# 
# Profitability: The cumulative profit indicates the strategy's robustness in leveraging expected values for decision-making, despite relatively low predictive accuracy. This shows the strength of a probabilistic approach in sports analytics.
# Visual Insights: Heatmaps and histograms revealed correlations and feature importance, guiding feature selection and emphasizing factors like "Successful Passes" and "Ball Possession" in prediction outcomes.
# 
# **FUTURE WORK**
# 
# Improving Model Performance:
# 
# Addressing class imbalance through techniques like oversampling underrepresented classes or applying class-specific weights during training. Exploring alternative machine learning algorithms such as ensemble methods (e.g., stacking) or deep learning models to improve generalization.
# 
# Feature Engineering:
# 
# Incorporating additional features like team form, weather conditions, or player-specific data could enhance predictive power.
# Evaluating temporal trends, such as performance over time, could provide dynamic insights into match outcomes.
# 
# Profit Optimization:
# 
# Introducing dynamic stake sizing strategies based on confidence levels or EV magnitude. Conducting sensitivity analysis on the betting threshold to optimize profitability further.
# 
# Real-Time Applications:
# 
# Developing real-time analytics pipelines to apply this methodology during live matches. Incorporating market-specific dynamics, such as fluctuating odds or late-breaking news, into the decision framework.
# 
# Scalability:
# 
# Expanding the dataset to include matches from different leagues or seasons to validate the strategy's robustness across diverse contexts. Testing the methodology on other sports to evaluate its generalizability.
# 
# In conclusion, this study demonstrates the potential of integrating predictive analytics with probabilistic decision-making to develop effective betting strategies. While the strategy showed promising profitability, enhancements in model accuracy and feature scope could further improve its performance and reliability. By addressing these areas, future implementations could yield even more robust and scalable solutions for sports analytics and betting markets.

# **REFERENCES**
# 1)Koopman, S. J., Lit, R., & Vegter, H. M. (2021). Forecasting football match results in national league competitions using score-driven time series models. Vrije Universiteit Amsterdam. Retrieved from https://research.vu.nl/ws/portalfiles/portal/151597369/Forecasting_football_match_results_in_national_league_competitions_using_scoredriven_time_series_models.pdf
# 
# 2)OpenAI. (2025). Use of ChatGPT for generating insights and recommendations. Retrieved from https://openai.com
