def train_model(model, X_train, y_train):
    """
    Fit model to training data.
    """
    model.fit(X_train, y_train)
    print("Model trained.")
    return model

def predict_model(model, X_test):
    """
    Predict labels for test data.
    """
    return model.predict(X_test)

def predict_proba_model(model, X_test):
    """
    Predict probabilities if model supports it.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:,1]
    else:
        print("Model does not support predict_proba.")
        return None
