# Data-Science-project-bankruptcy-in-poland-Imbalanced-Data
# Bankruptcy in Poland Data Science Project

This project aims to predict bankruptcy in Polish companies using machine learning algorithms. The dataset used for this project is the "Bankruptcy in Poland" dataset, which contains financial ratios and other financial indicators for a sample of Polish companies.

## Data Preparation

The first step in our data preparation is to load the arff file into a pandas dataframe. We then rename the class column to "bankrupt" to make it more understandable, and replace the values with 0 (not bankrupt) and 1 (bankrupt) to make it easier for predicting.

We also see some issues with multicollinearity in our exploratory data analysis (EDA). This means that some of our features are highly correlated with each other, which can cause problems for our models.

Our data is also imbalanced, meaning that we have more examples of non-bankrupt companies than bankrupt ones. To address this issue, we need to balance our data during preparation.

Finally, many of our features have missing values that we'll need to impute. Since the features are highly skewed, the best imputation strategy is likely median rather than mean.

## Model Training

We divide the data into training and test sets using a randomized train-test split. One strategy we use is resampling the training data to balance our classes.

Note that because our classes are imbalanced, the baseline accuracy is very high. However, this doesn't necessarily mean that it's actually good.

We create a model using SimpleImputer and DecisionTreeClassifier algorithms. We then evaluate its accuracy on both training and test sets and plot a confusion matrix.

## Conclusion

This project shows how machine learning algorithms can be used to predict bankruptcy in Polish companies based on financial indicators. We demonstrate how balancing classes during preparation and avoiding linear models can improve model performance.
