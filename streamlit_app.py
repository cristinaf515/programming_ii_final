#!/usr/bin/env python
# coding: utf-8

# # Programming II Final
# ## Cristina Ferreira
# ### 12/09/25

# ***

# #### Q1

# In[1]:


### Step 0: Setup ----

import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


# Load data

s = pd.read_csv("social_media_usage.csv")

# Check dimensions of dataset
s.shape


# ***

# #### Q2

# In[3]:


# define toy dataframe

def clean_sm(x):
    return np.where(x==1,1,0)
    
toy_df = pd.DataFrame({
    "col1" : [1,2,3],
    "col2" : [2,1,0]
})

toy_df

toy_clean=toy_df.apply(clean_sm)
toy_clean


# ***

# #### Q3

# In[4]:


s["sm_li"] = clean_sm(s["web1h"])

ss = pd.DataFrame({
    "sm_li": s["sm_li"],
    "income": np.where((s["income"] >= 1) & (s["income"] <= 9), s["income"], np.nan),
    "education": np.where((s["educ2"] >= 1) & (s["educ2"] <= 8), s["educ2"], np.nan),
    "parent": np.where(s["par"] == 1, 1, np.where(s["par"] == 2, 0, np.nan)),
    "married": np.where(s["marital"] == 1, 1, np.where((s["marital"] >= 2) & (s["marital"] <= 6), 0, np.nan)),
    "female": np.where(s["gender"] == 2, 1, np.where(s["gender"] == 1, 0, np.nan)),
    "age": np.where((s["age"] >= 0) & (s["age"] <= 98), s["age"], np.nan)
})

# Drop missing values
ss = ss.dropna()


# ***

# In[6]:


# Create labeled version of sm_li for plotting
ss["sm_li_lab"] = ss["sm_li"].map({0: "No (0)", 1: "Yes (1)"}).astype("category")

# Summary statistics
print("\nSummary Statistics:")
print(ss.describe())

# Average LinkedIn use across numeric variables
print("\nLinkedIn Users across numeric variables:")
print(ss.groupby("sm_li")[["income", "education", "age"]].mean())

# Boxplots for numeric predictors
sns.boxplot(data=ss, x="sm_li_lab", y="income", hue="sm_li_lab")
plt.title("Income by LinkedIn Use")
plt.legend(title="LinkedIn Use")
plt.show()

sns.boxplot(data=ss, x="sm_li_lab", y="education", hue="sm_li_lab")
plt.title("Education by LinkedIn Use")
plt.legend(title="LinkedIn Use")
plt.show()

sns.boxplot(data=ss, x="sm_li_lab", y="age", hue="sm_li_lab")
plt.title("Age by LinkedIn Use")
plt.legend(title="LinkedIn Use")
plt.show()


# Average LinkedIn use across binary variables
print("\nLinkedIn Users across binary variables:")
print(ss.groupby("sm_li")[["parent", "married", "female"]].mean())

# Bar plots for binary predictors
sns.barplot(data=ss, x="sm_li_lab", y="parent", hue="sm_li_lab")
plt.title("Parent Status by LinkedIn Use")
plt.legend(title="LinkedIn Use")
plt.show()

sns.barplot(data=ss, x="sm_li_lab", y="married", hue="sm_li_lab")
plt.title("Married Status by LinkedIn Use")
plt.legend(title="LinkedIn Use")
plt.show()

sns.barplot(data=ss, x="sm_li_lab", y="female", hue="sm_li_lab")
plt.title("Female by LinkedIn Use")
plt.legend(title="LinkedIn Use")
plt.show()


# Crosstabs for binary variables

print("\nCrosstab: Parent by LinkedIn Use")
print(pd.crosstab(ss["sm_li"], ss["parent"], normalize="index"))

print("\nCrosstab: Married by LinkedIn Use")
print(pd.crosstab(ss["sm_li"], ss["married"], normalize="index"))

print("\nCrosstab: Female by LinkedIn Use")
print(pd.crosstab(ss["sm_li"], ss["female"], normalize="index"))


# In[7]:


# Pairplot of key variables
predictors = ["income", "education", "parent", "married", "female", "age"]

eda_df = ss[predictors + ["sm_li"]].copy()

# pairplot
g = sns.pairplot(
    eda_df,
    hue="sm_li",
    diag_kind="kde",
    plot_kws={"alpha": 0.6}
)

g.fig.suptitle("Exploratory Pairplot of Key Variables by LinkedIn Usage", y=1.02)
plt.show()


# #### Q4

# In[8]:


# Target vector (y)
y = ss["sm_li"]

# Feature set (x)
x = ss[["income", "education", "parent", "married", "female", "age"]]


# ***

# #### Q5 

# In[9]:


# train and test sets

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=456)


# Machine learning works by training a model on one set of data and then checking how well it performs on new data it hasn’t seen before. In our case, X_train contains 80% of the feature rows and is used to teach the model the relationship between the predictors (household income, education level, if a parent or not, marital status, female gender, and age) and LinkedIn use. X_test holds the remaining 20% of the rows and is not shown to the model during training. This set is used only to judge how well the model performs on new, future cases.
# 
# y_train holds the LinkedIn usage labels (1 = uses LinkedIn, 0 = does not use LinkedIn) for the rows in X_train, and allows the model to learn patterns between the features and the target. y_test contains the true LinkedIn usage labels for the test rows and is used to compare the model’s predictions with the actual outcomes. This tells us how well the model would predict LinkedIn use for future users.

# ***

# #### Q6

# In[9]:


# Logistic regression

log_reg = LogisticRegression(
    class_weight="balanced", 
    random_state=51598, 
    solver="liblinear"
)

# Fit the model on the training data
log_reg.fit(x_train,y_train)


# ***

# #### Q7

# In[10]:


# Evaluate model using test data
y_pred = log_reg.predict(x_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)


# In[11]:


# Confusion matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# The confusion matrix explains how well the logistic regression model differentiate LinkedIn users from non-users. 110 represents the true negatives, or the number of people who do not use LinkedIn that the model correctly predicted. 60 represents false positives, or the number of people that do not use LinkedIn but the model incorrectly identified as users. 19 represents  false negatives, or the number of LinkedIn users that the model failed to identify. Lastly, 60 represents the true positives, or the number of LinkedIn users correctly predicted by the model.

# ***

# #### Q8

# In[12]:


# Confusion matrix on dataframe

confusion_matrix_df = pd.DataFrame(
    cm,
    columns=["Predicted Non-User (0)", "Predicted User (1)"],
    index=["Actual Non-User (0)", "Actual User (1)"]
)

print(confusion_matrix_df)


# ***

# #### Q9

# In[13]:


# Precision, recall, and F1 score

cm2 = confusion_matrix(y_test, y_pred)

tn = cm2[0, 0]
fp = cm2[0, 1]
fn = cm2[1, 0]
tp = cm2[1, 1]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# The precision score measures how often the model correctly predicts someone is a LinkedIn user. The model is correct 50% of the time when it predicts a true positive value. High precision is a valuable metric when, for example, a company pays for expensive LinkedIn advertising and wants someone to actually target only LinkedIn users instead of wasting money on non-LinkedIn users (false positives). In this case, precision is not ideal as it means the model predicts LinkedIn users accurately only half of the time, and incorrectly predicts LinkedIn users the other half of the time.

# The recall measures how many actual LinkedIn users that the model successfully identified. The model identifies approximately 76% of actual LinkedIn users. A high recall is valuable if, for example, a company wants to ensure their advertisement reaches actual LinkedIn users (true positives) and doesn't reach non-LinkedIn users (false positives).

# The F1 score combines precision and recall to balance is the model can correctly identify LinkedIn users (recall) and not incorrectly label non-users as users (precision). An F1 score of approximately 60% means the model performs decently well, but the model is better at identifying LinkedIn users (high recall) than it is being correct when predicting someone is a user (low precision).

# In[14]:


print(classification_report(y_test, y_pred, target_names=["Non-user (0)", "LinkedIn user (1)"]))


# ***

# #### Q10

# In[15]:


# Predictions

predictions = pd.DataFrame({
    "income":   [8, 8],   # high income
    "education":[7, 7],   # high education
    "parent":   [0, 0],   # non-parent
    "married":  [1, 1],   # married
    "female":   [1, 1],   # female
    "age":      [42, 82]  # age changes
})

predictions

probability = log_reg.predict_proba(predictions)
probability


# In[16]:


# Column 0 --> probability does NOT use LinkedIn, column 1 --> probability USES LinkedIn

linkedin_probs = probability[:, 1]
linkedin_probs


# The person that is 42 years old has a approximately 69% predicted probability of using LinkedIn.
# The person that is 82 years old has a approximately 45% predicted probability of using LinkedIn.
# Both individuals have a high income, are highly educated, are not parents, and are married females. These results indicate that age and LinkedIn usage have a negative relationship; holding all other factors constant, younger people are more likely to use LinkedIn compared to older people.

# #### Part 2: Streamlit

# In[ ]:


model = log_reg                 # reuse your trained model
test_acc = accuracy             # reuse the accuracy you already computed
test_cm = cm                    # reuse the confusion matrix

test_report = classification_report(
    y_test,
    y_pred,
    target_names=["Non-user (0)", "LinkedIn user (1)"]
)



# In[ ]:


import streamlit as st

# define function for streamlit so above dataframes aren't pulled into
def run_app():
    st.title("Predicting LinkedIn Usage")
    ...

    
# Reuse your trained model
model = log_reg
test_acc = accuracy
test_cm = cm
test_report = classification_report(
    y_test,
    y_pred,
    target_names=["Non-user (0)", "LinkedIn user (1)"]
)

st.title("Predicting LinkedIn Usage")
st.write(
    "This app uses a logistic regression model to predict whether someone uses LinkedIn "
    "based on their income, education, parental status, marital status, gender, and age."
)


# Sidebar for app users to select individual characteristics
st.sidebar.header("Input: Individual Characteristics")

income = st.sidebar.slider("Household income (1 = lowest, 9 = highest)", 1, 9, 5)
education = st.sidebar.slider("Education (1 = less than HS, 8 = post-grad)", 1, 8, 4)

parent_str = st.sidebar.radio("Is the person a parent?", ["No", "Yes"])
parent = 1 if parent_str == "Yes" else 0

married_str = st.sidebar.radio("Is the person married?", ["No", "Yes"])
married = 1 if married_str == "Yes" else 0

female_str = st.sidebar.radio("Gender", ["Male", "Female"])
female = 1 if female_str == "Female" else 0

age = st.sidebar.slider("Age", 18, 98, 35)

# Prediction button

if st.button("Predict LinkedIn Use"):

    # build one-row dataframe in the same order as training
    input_df = pd.DataFrame({
        "income":   [income],
        "education":[education],
        "parent":   [parent],
        "married":  [married],
        "female":   [female],
        "age":      [age]
    })

    # 1) predicted LinkedIn usage (0 = non-user, 1 = user)
    y_pred_single = model.predict(input_df)[0]

    # 2) probabilities for LinkedIn usage (0 = non-user, 1 = user)
    proba_nonuser, proba_user = model.predict_proba(input_df)[0]

    st.subheader("Prediction for this person")

    # if prediction is true (1), write: 
    if y_pred_single == 1:
        st.write("**Classification:** This person is predicted to be a **LinkedIn user (1)**.")
    # if prediction is false (0), write: 
    else:
        st.write("**Classification:** This person is predicted to be a **Non-user (0)**.")

    # include probabilities
    st.write(
        f"**Probability they use LinkedIn:** {proba_user:.1%}  "
        f"(and {proba_nonuser:.1%} probability they do **not** use LinkedIn)."
    )

    # visual
    proba_df = pd.DataFrame(
        {
            "Status": ["Non-user", "LinkedIn user"],
            "Probability": [proba_nonuser, proba_user],
        }
    ).set_index("Status")

    st.bar_chart(proba_df)


    # model performance (accuracy, confusion matrix)
    st.subheader("Model performance (overall, not this person)")

    st.write(f"Accuracy on test set: **{test_acc:.3f}**")

    confusion_matrix_df = pd.DataFrame(
        test_cm,
        columns=["Predicted Non-User (0)", "Predicted User (1)"],
        index=["Actual Non-User (0)", "Actual User (1)"]
    )

    with st.expander("Show confusion matrix and full classification report"):
        st.write(confusion_matrix_df)
        st.text(test_report)

