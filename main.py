
import data_loader
import preprocessing
import model_selection
import evaluation
import prediction

def main():
    data = data_loader.load_data("loan_data.xlsx")
    X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = preprocessing.preprocess_data(data)
    
    models = model_selection.train_models(X_train, y_train, preprocessor)
    
    results = evaluation.evaluate_models(models, X_test, y_test)
    for model_name, metrics in results.items():
        print(f"\n{model_name} Evaluation Results:")
        for metric, value in metrics.items():
            if metric == "Confusion Matrix" or metric == "Classification Report":
                print(f"{metric}:\n{value}")
            else:
                print(f"{metric}: {value:.2f}")
    
    if 'Logistic Regression' in models:
        logistic_pipeline = models['Logistic Regression']
        equation = prediction.get_logistic_equation(logistic_pipeline, numerical_features, categorical_features)
        print("\nLogistic Regression Equation:", equation)

if __name__ == "__main__":
    main()
