import joblib

def save_model(model, version):
    joblib.dump(model, f'models/decision_tree_v{version}.pkl')

def load_model(version):
    return joblib.load(f'models/decision_tree_v{version}.pkl')