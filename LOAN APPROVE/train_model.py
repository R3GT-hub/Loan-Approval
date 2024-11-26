import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
def preprocess_data():
    # Read data
    data = pd.read_csv("loan_approval_dataset.csv")
    
    # Drop loan_id and combine asset columns
    data.drop(columns=["loan_id"], inplace=True)
    data.columns = data.columns.str.strip()
    
    # Calculate total assets
    data["Assets"] = (
        data.residential_assets_value +
        data.commercial_assets_value +
        data.luxury_assets_value +
        data.bank_asset_value
    )
    
    # Drop individual asset columns
    data.drop(
        columns=[
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ],
        inplace=True,
    )
    
    # Clean and encode categorical variables
    data.education = data.education.str.strip()
    data.self_employed = data.self_employed.str.strip()
    data.loan_status = data.loan_status.str.strip()
    
    data["education"] = data["education"].replace(["Graduate", "Not Graduate"], [1, 0])
    data["self_employed"] = data["self_employed"].replace(["No", "Yes"], [0, 1])
    data["loan_status"] = data["loan_status"].replace(["Approved", "Rejected"], [1, 0])
    
    return data

def evaluate_model(model, X_test, y_test, X_test_scaled):
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': abs(model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()
    
    return accuracy, precision, recall, f1

def main():
    # Preprocess data
    data = preprocess_data()
    
    # Split features and target
    X = data.drop(columns=["loan_status"])
    y = data["loan_status"]
    
    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, X_test_scaled)
    
    # Save model and scaler
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    
    # Save metrics to file
    with open('model_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {metrics[0]:.4f}\n")
        f.write(f"Precision: {metrics[1]:.4f}\n")
        f.write(f"Recall: {metrics[2]:.4f}\n")
        f.write(f"F1 Score: {metrics[3]:.4f}\n")

if __name__ == "__main__":
    main() 