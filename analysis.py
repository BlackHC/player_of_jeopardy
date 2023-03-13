"""
Plot the results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datasets

# Set a nice seaborn style for matplotlib
sns.set_theme()


#%%

# Load results from csv
df = pd.read_csv("jeopardy_results.csv", index_col="idx")

#%%
# Load the dataset from the Hugging Face Hub
dataset = datasets.load_dataset("jeopardy", split="train")

# Turn dataset into a dataframe
dataset = pd.DataFrame(dataset)

# Rename the category column to avoid conflicts
dataset.rename(columns={"category": "category_dataset", "question": "question_dataset"}, inplace=True)

#%%

# Join the dataset with the results (we don't have results for all rows)
full_df = df.join(dataset, how="inner")

# Verify that category_dataset and category are the same
assert (full_df["category_dataset"] == full_df["category"]).all()
# Verify that question_dataset and question are the same
assert (full_df["question_dataset"] == full_df["question"]).all()

# Delete category_dataset and question_dataset
del full_df["category_dataset"]
del full_df["question_dataset"]

#%%

# We have one nan
# The log message is: Expected confidence between 0 and 1, got content='I apologize, but I cannot provide a specific numerical value of my confidence level, as I am an artificial intelligence language model, and I do not have personal feelings or emotions. However, based on my knowledge and analysis of the available information, I am confident that my answer (South Africa) is correct.' additional_kwargs={}
# Check that that is the case
assert len(full_df[full_df["confidence"].isna()]) == 1
assert full_df[full_df["confidence"].isna()].iloc[0]["answer"] == "South Africa"
# Set the confidence to 1.
full_df["confidence"].fillna(1, inplace=True)

#%%

# Plot the distribution of confidence
sns.histplot(data=full_df, x="confidence", bins=20)
plt.show()

#%%

# Plot a calibration plot using sklearn
from sklearn.calibration import CalibrationDisplay

# Get the calibration display
cal_display = CalibrationDisplay.from_predictions(
    y_true=full_df["accuracy"], y_prob=full_df["confidence"], n_bins=5, name="ChatGPT",
    strategy="uniform"
)
# Plot the calibration curve
cal_display.plot()
plt.savefig("chatgpt_calibration.svg", format="svg", bbox_inches="tight", pad_inches=0, transparent=False)
plt.show()


#%%

# Plot the AUROC curve with RocCurveDisplay
from sklearn.metrics import RocCurveDisplay

roc_display = RocCurveDisplay.from_predictions(
    y_true=full_df["accuracy"], y_pred=full_df["confidence"], name="ChatGPT")
# Plot the ROC curve
roc_display.plot()
plt.show()

#%% Load the watson_cmp data
import numpy as np

watson_cmp = pd.read_csv("watson_cmp/watson_v0.8_precision_recall.csv")

# Sort the data by recall (ascending)
watson_cmp.sort_values(by="recall", inplace=True)

# Compute the average precision score for watson_cmp (which has recall, precision as columns)
# Use np.sum(np.diff(recall) * np.array(precision)[:-1]) to compute the area under the curve
watson_avg_precision = np.sum(np.diff(watson_cmp["recall"]) * np.array(watson_cmp["precision"])[:-1])
print(f"watson_avg_precision: {watson_avg_precision}")

#%%

# Plot the precision-recall curve with PrecisionRecallDisplay
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.ticker as mtick

pr_display = PrecisionRecallDisplay.from_predictions(
    y_true=full_df["accuracy"], y_pred=full_df["confidence"], name="ChatGPT")
# Plot the precision-recall curve
pr_display.plot()

pr_display_watson = PrecisionRecallDisplay(
    precision=watson_cmp["precision"], recall=watson_cmp["recall"],
    average_precision=watson_avg_precision,
    estimator_name="Watson v0.8"
)
# Plot the precision-recall curve for Watson
pr_display_watson.plot(ax=plt.gca())

# X axis is % Answered
plt.xlabel("% Answered")
# Change the ticks and labels to be percentages (in 10% increments)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
           ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])

# Y axis is Precision
plt.ylabel("Precision")
# Change the labels to be in percentages
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.savefig("chatgpt_watson_v0.8_precision_recall.svg", format="svg", bbox_inches="tight", pad_inches=0, transparent=False)
plt.show()

#%% Compute accuracy by round

# Get the number of correct answers by round
correct_by_round = full_df.groupby(["round"]).agg({"accuracy": "sum"})

# Get the total number of answers by round
total_by_round = full_df.groupby(["round"]).agg({"accuracy": "count"})

# Compute the accuracy by round
accuracy_by_round = correct_by_round / total_by_round

# Render the accuracy by round as markdown table
print(accuracy_by_round.to_markdown())

#%%



