from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# True labels and predicted labels from a classification model
y_true = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]

# Calculate precision
precision = precision_score(y_true, y_pred)
print(f'Precision: {precision}')

# Calculate recall
recall = recall_score(y_true, y_pred)
print(f'Recall: {recall}')

# Calculate F1-score
f1 = f1_score(y_true, y_pred)
print(f'F1-score: {f1}')