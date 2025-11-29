from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np

# predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # for PR/thresholds

# 1) classification report
print(classification_report(y_test, y_pred, digits=4))

# 2) confusion matrix plot
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix'); plt.show()

# 3) macro/micro/weighted f1
from sklearn.metrics import f1_score
print('F1 macro:', f1_score(y_test, y_pred, average='macro'))
print('F1 micro:', f1_score(y_test, y_pred, average='micro'))
print('F1 weighted:', f1_score(y_test, y_pred, average='weighted'))

# 4) per-class PR curves (one-vs-rest)
from sklearn.preprocessing import label_binarize
classes = sorted(list(set(y_test)))
Y_test_bin = label_binarize(y_test, classes=classes)
for i, cls in enumerate(classes):
    if y_proba.shape[1] == len(classes):
        scores = y_proba[:, i]
    else:
        # fallback: probability not available
        continue
    precision, recall, _ = precision_recall_curve(Y_test_bin[:, i], scores)
    ap = average_precision_score(Y_test_bin[:, i], scores)
    plt.plot(recall, precision, label=f'{cls} (AP={ap:.3f})')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.title('PR curves'); plt.show()
