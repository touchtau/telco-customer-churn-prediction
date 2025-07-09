from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def build_random_forest(**kwargs):
    """
    Create random forest classifier.
    """
    return RandomForestClassifier(**kwargs)

def build_xgboost(**kwargs):
    """
    Create XGBoost classifier.
    """
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
