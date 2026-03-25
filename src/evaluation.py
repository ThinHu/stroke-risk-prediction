import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score

def evaluate_and_plot(model, X_test, y_test, model_name="Model", threshold=None):
    """
    Prints metrics and plots a confusion matrix.
    If threshold is provided, it uses predict_proba() to apply the custom threshold.
    Otherwise, it defaults to standard predict().
    """
    # Xử lý Threshold tùy chỉnh nếu có
    if threshold is not None and hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        display_name = f"{model_name} (Threshold {threshold})"
    else:
        # Fallback về mặc định (dùng cho các model gọi hàm không có tham số threshold)
        y_pred = model.predict(X_test)
        display_name = model_name
    
    print(f"\n--- {display_name.upper()} RESULTS ---")
    print(f"Recall (Stroke): {recall_score(y_test, y_pred):.2f}")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens' if 'Stacking' in model_name else 'Blues')
    plt.title(f'Confusion Matrix - {display_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def simulate_thresholds(y_test, y_probs, thresholds=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5]):
    """Simulates performance across different probability thresholds."""
    print("\n--- THRESHOLD TUNING SIMULATION ---")
    print(f"{'Threshold':<10} | {'Recall':<20} | {'False Alarms (FP)':<20}")
    print("-" * 55)
    
    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        recall = recall_score(y_test, y_pred_t)
        
        # Calculate False Positives from confusion matrix
        cm = confusion_matrix(y_test, y_pred_t)
        false_alarms = cm[0][1]
        
        print(f"{t:<10.2f} | {recall:<20.2f} | {false_alarms:<20}")