from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_model(df, target):
    # Split features and target
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    
    # Accuracy
    print("KNN Accuracy:", accuracy_score(y_test, preds))