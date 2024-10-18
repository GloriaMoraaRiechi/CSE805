from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef

# True labels and predicted labels from a classification model
y_true = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]  # Actual labels
y_pred = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]  # Predicted labels

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() #unpack the confusion matrix into its individual components for further calculations

print(f'True Positives (TP): {tp}')
print(f'True Negatives (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')

# Calculate Precision, Recall, and F1-score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f"Matthews Correlation Coefficient: {mcc:.3f}")

