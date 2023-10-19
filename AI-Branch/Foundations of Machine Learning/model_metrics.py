from sklearn.metrics import confusion_matrix, classification_report

# Assuming you've built a classification model and got y_pred and y_test
# For this example, dummy values for y_pred and y_test
y_test = [1, 0, 1, 1, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1]

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Get precision, recall, and F1 score
report = classification_report(y_test, y_pred)
print(report)
