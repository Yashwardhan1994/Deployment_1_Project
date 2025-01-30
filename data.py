import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , MinMaxScaler , StandardScaler
from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV , RandomizedSearchCV , cross_validate
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report , recall_score  , precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier , AdaBoostClassifier , GradientBoostingClassifier , RandomForestClassifier , VotingClassifier
from xgboost import XGBClassifier
from scipy.stats import randint
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\Yash\OneDrive\Desktop\GLIM\Term 5\ML-2\Project\bank-full.csv" , sep=';')
data.head()
data.value_counts("y")
data.dtypes

data = data.rename(columns={'y': 'Subscibed'})
import numpy as np

# Replace 'unknown' and other placeholders with NaN
data.replace(['unknown','?'], np.nan, inplace=True)


for column in data.columns:
    print(f"Feature: {column}")
    print(data[column].value_counts())
    print("\n" + "-"*10 + "\n")

data.isnull().sum()

data.dtypes

#Loop for plotting all the histogram of all the integer features
data.hist(bins=10, color='silver', edgecolor='black', linewidth=1.0,
              xlabelsize=10, ylabelsize=10, grid=False)    
plt.tight_layout(rect=(0, 0, 2, 2))   
rt = plt.suptitle('Univariate Histogram Plots for all Int Features', x=0.85, y=2, fontsize=8)

#Loop for plotting all the count plot of all the Category and object features.
for column in data.select_dtypes(include=['object', 'category']).columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=column, palette='Set2')
    plt.title(f'Count Plot for {column.capitalize()}')
    plt.xlabel(column.capitalize())
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
    plt.show()


#Loop for plotting all the count plot of all the Category and object features.
for column in data.select_dtypes(include=['int', 'float']).columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='Subscibed', y=column, palette='Set1')
    plt.title(f'Count Plot for {column.capitalize()}')
    plt.xlabel(column.capitalize())
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# List of variables to plot
variables = ['age', 'day', 'campaign', 'previous']

# Set up the figure
plt.figure(figsize=(60, 24))

# Loop through the variables and create subplots
for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)  # Dynamically assign subplots in a 2x2 grid
    sns.countplot(x=var, hue='Subscibed', data=data, palette="husl")
    plt.title(f"{var.capitalize()} vs Subscribed", fontsize=24)  # Dynamic title
    plt.xlabel(var.capitalize(), fontsize=18)  # Dynamic x-axis label
    plt.ylabel("Count", fontsize=18)
    plt.xticks(fontsize=18)  # Adjust x-axis tick font size
    plt.yticks(fontsize=18)  # Adjust y-axis tick font size

# Add legend and adjust layout
plt.legend(title='Subscribed', fontsize=20, title_fontsize=16, loc='upper right')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Define the pairs of x and y for scatterplots
plot_pairs = [
    ('age', 'balance'),
    ('balance', 'day'),
    ('balance', 'duration'),
    ('duration', 'age'),
    ('campaign', 'pdays'),
    ('pdays', 'previous'),
    ('previous', 'balance')
]

# Set up the figure
plt.figure(figsize=(20, 12))

# Loop through the plot pairs and create scatterplots
for i, (x, y) in enumerate(plot_pairs, 1):
    plt.subplot(2, 4, i)  # Create subplots (2 rows, 4 columns)
    sns.scatterplot(x=x, y=y, hue='Subscibed', data=data, palette="Set1")
    plt.title(f"{x} vs {y}")  # Dynamic title

# Adjust layout for better spacing
plt.tight_layout()

# Display the plots
plt.show()

import plotly.express as px

# Define the variable combinations for 3D scatter plots
plot_combinations = [
    ('age', 'balance', 'day'),
    ('balance', 'day', 'duration'),
    ('day', 'duration', 'age'),
    ('duration', 'age', 'campaign'),
    ('campaign', 'pdays', 'previous'),
    ('pdays', 'previous', 'balance')
]

# Loop through the combinations and create 3D scatter plots
for x, y, z in plot_combinations:
    fig = px.scatter_3d(
        data,
        x=x,
        y=y,
        z=z,
        color='Subscibed',  # Grouping by 'Subscibed'
        title=f"3D Scatter Plot of {x} vs {y} vs {z}",
        labels={x: x.capitalize(), y: y.capitalize(), z: z.capitalize()}  # Dynamic labels
    )
    # Show the plot
    fig.show()

mode = X_train[['job', 'poutcome', 'contact', 'education']].dropna().mode().iloc[0]
print(mode)
X_test['job'].fillna(mode , inplace = True)
X_train['job'].fillna(mode , inplace = True)
X_test['contact'].fillna(mode , inplace = True)
X_train['contact'].fillna(mode , inplace = True)
median = X_train['balance'].dropna().median()
print(median)
X_test['balance'].fillna(median, inplace = True)
X_train['balance'].fillna(median, inplace = True)
labelEncoder= LabelEncoder()

X_train['education'] = labelEncoder.fit_transform(X_train['education'])
X_test['education'] = labelEncoder.transform(X_test['education'])

y_train = labelEncoder.fit_transform(y_train)

y_test = labelEncoder.transform(y_test)

categorical_columns = X.select_dtypes(include=['category', 'object']).columns

# Create and fit OneHotEncoder on the training data
ohe = OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore')

# Fit and transform on the training data (will create sparse matrix)
X_train_cat_sparse = ohe.fit_transform(X_train[categorical_columns])

# Transform the test data using the same encoder (sparse matrix as well)
X_test_cat_sparse = ohe.transform(X_test[categorical_columns])

# Convert back to DataFrame for readability (converting sparse to dense format for visualization purposes)

# It's optional to do this step, as the model will work with sparse format
X_train_cat = pd.DataFrame.sparse.from_spmatrix(X_train_cat_sparse, columns=ohe.get_feature_names_out(categorical_columns), index=X_train.index)
X_test_cat = pd.DataFrame.sparse.from_spmatrix(X_test_cat_sparse, 
columns=ohe.get_feature_names_out(categorical_columns), index=X_test.index)

# Drop original categorical columns from X_train and X_test
X_train = X_train.drop(columns=categorical_columns)
X_test = X_test.drop(columns=categorical_columns)

# Concatenate the encoded categorical columns back (sparse format will be preserved)
X_train = pd.concat([X_train, X_train_cat], axis=1)
X_test = pd.concat([X_test, X_test_cat], axis=1)

# Check the shapes to ensure alignment
print(X_train.shape, X_test.shape)

# print(set(y_train))
X_train.columns
#Features to scale

Numerical_columns = X.select_dtypes(include=['int']).columns

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on training data and transform
X_train[Numerical_columns] = scaler.fit_transform(X_train[Numerical_columns])

# Transform the test data using the same scaler
X_test[Numerical_columns] = scaler.transform(X_test[Numerical_columns])
X_train.head()
# Define and instantiate the models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=25),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=25),
    'VotingClassifier': VotingClassifier(estimators=[
        ('dt1', DecisionTreeClassifier(criterion="entropy", random_state=25)),
        ('dt2', DecisionTreeClassifier(random_state=25)),
        ('dt3', DecisionTreeClassifier(random_state=25))
    ], voting='soft'),
    'BaggingClassifier': BaggingClassifier(random_state=25),
    'AdaBoostClassifier': AdaBoostClassifier(random_state=25),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=25),
    'RandomForestClassifier': RandomForestClassifier(random_state=25),
    'XGBClassifier': XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    'SVClassifier': SVC(random_state=25)
}
def train_classifier(model, X_train, y_train, X_test, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Predictions for train and test datasets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # Calculate recall and precision on the test set
    recall = recall_score(y_test, y_pred_test )  
    precision = precision_score(y_test, y_pred_test) 

    return accuracy_test, accuracy_train, recall, precision
# Lists to store metrics for all models
accuracy_scores_test = []
accuracy_scores_train = []
precision_scores = []
recall_scores = []

# Loop through all models
for name, model in models.items():
    current_accuracy_test, current_accuracy_train, current_recall, current_precision = train_classifier(
        model, X_train, y_train, X_test, y_test
    )

    # Print metrics for the current model
    print(f"\nFor model: {name}")
    print(f"Test Accuracy: {current_accuracy_test:.4f}")
    print(f"Train Accuracy: {current_accuracy_train:.4f}")
    print(f"Recall (Test): {current_recall:.4f}")
    print(f"Precision (Test): {current_precision:.4f}")

    # Append metrics to their respective lists
    accuracy_scores_test.append(current_accuracy_test)
    accuracy_scores_train.append(current_accuracy_train)
    precision_scores.append(current_precision)
    recall_scores.append(current_recall)

# # Summary of metrics
# print("\nSummary of metrics for all models:")
# for i, name in enumerate(models.keys()):
#     print(f"Model: {name}")
#     print(f"  Test Accuracy: {accuracy_scores_test[i]:.4f}")
#     print(f"  Train Accuracy: {accuracy_scores_train[i]:.4f}")
#     print(f"  Recall: {recall_scores[i]:.4f}")
#     print(f"  Precision: {precision_scores[i]:.4f}")
# Plot metrics
def plot_metrics(models, accuracy_scores_test, accuracy_scores_train, precision_scores, recall_scores):
    # Set the bar width
    bar_width = 0.2
    # Positions of bars on the x-axis
    indices = np.arange(len(models))
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Bar plots for each metric
    plt.bar(indices, accuracy_scores_test, bar_width, label='Test Accuracy')
    plt.bar(indices + bar_width, accuracy_scores_train, bar_width, label='Train Accuracy')
    plt.bar(indices + 2 * bar_width, precision_scores, bar_width, label='Precision')
    plt.bar(indices + 3 * bar_width, recall_scores, bar_width, label='Recall')
    
    # Labels and title
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Metrics Comparison Across Models', fontsize=16)
    plt.xticks(indices + 1.5 * bar_width, list(models.keys()), rotation=45, ha='right', fontsize=12)
    plt.legend()
    
    # Add grid for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Call the function to plot
plot_metrics(models, accuracy_scores_test, accuracy_scores_train, precision_scores, recall_scores)