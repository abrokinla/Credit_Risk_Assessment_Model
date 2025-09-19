"""
evaluate.py
Model evaluation, metrics, and explainability.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_classification_model(model, X, y_true):
	"""
	Evaluate a classification model and return metrics as a dict.
	"""
	y_pred = model.predict(X)
	if hasattr(model, "predict_proba"):
		y_proba = model.predict_proba(X)[:, 1]
	else:
		y_proba = None
	results = {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred, zero_division=0),
		"recall": recall_score(y_true, y_pred, zero_division=0),
		"f1": f1_score(y_true, y_pred, zero_division=0),
		"confusion_matrix": confusion_matrix(y_true, y_pred),
		"classification_report": classification_report(y_true, y_pred, zero_division=0)
	}
	if y_proba is not None:
		results["roc_auc"] = roc_auc_score(y_true, y_proba)
	return results

# Optional: add LIME/SHAP explainability hooks here
