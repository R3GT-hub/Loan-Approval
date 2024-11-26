import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Import different models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def preprocess_data():
    # Same preprocessing as before
    data = pd.read_csv("loan_approval_dataset.csv")
    data.drop(columns=["loan_id"], inplace=True)
    data.columns = data.columns.str.strip()
    
    data["Assets"] = (
        data.residential_assets_value +
        data.commercial_assets_value +
        data.luxury_assets_value +
        data.bank_asset_value
    )
    
    data.drop(
        columns=[
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ],
        inplace=True,
    )
    
    data.education = data.education.str.strip()
    data.self_employed = data.self_employed.str.strip()
    data.loan_status = data.loan_status.str.strip()
    
    data["education"] = data["education"].replace(["Graduate", "Not Graduate"], [1, 0])
    data["self_employed"] = data["self_employed"].replace(["No", "Yes"], [0, 1])
    data["loan_status"] = data["loan_status"].replace(["Approved", "Rejected"], [1, 0])
    
    return data

def evaluate_model(model, X_test, y_test, X_test_scaled):
    y_pred = model.predict(X_test_scaled)
    
    # Calculate basic metrics without importing sklearn.metrics
    accuracy = np.mean(y_pred == y_test)
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_and_evaluate_models():
    # Prepare data
    data = preprocess_data()
    X = data.drop(columns=["loan_status"])
    y = data["loan_status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models (removed SVM and XGBoost)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test, y_test, X_test_scaled)
        results[name] = metrics
        
        # Create models directory if it doesn't exist
        import os
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Save each model
        pickle.dump(model, open(f"models/{name.lower().replace(' ', '_')}.pkl", "wb"))
        
        # Track best model
        if metrics['accuracy'] > best_score:
            best_score = metrics['accuracy']
            best_model = (name, model)
    
    # Save scaler
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    
    # Save best model separately
    pickle.dump(best_model[1], open("models/best_model.pkl", "wb"))
    
    return results, best_model[0]

def plot_comparison(results):
    # Prepare data for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        metric_values = [results[model][metric] for model in models]
        
        # Create bar plot
        sns.barplot(x=models, y=metric_values)
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Create detailed comparison table
    comparison_df = pd.DataFrame(results).round(4)
    comparison_df.to_csv('model_comparison.csv')

def main():
    # Train and evaluate all models
    results, best_model = train_and_evaluate_models()
    
    # Plot comparisons
    plot_comparison(results)
    
    # Print results
    print("\nModel Performance Summary:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    print(f"\nBest performing model: {best_model}")
    
    # Save results to text file
    with open('model_comparison.txt', 'w') as f:
        f.write("Model Performance Summary:\n")
        for model, metrics in results.items():
            f.write(f"\n{model}:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        f.write(f"\nBest performing model: {best_model}")

if __name__ == "__main__":
    main() 