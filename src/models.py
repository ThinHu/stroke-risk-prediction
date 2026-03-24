from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_base_models():
    """Returns the base classifiers used for ensembles."""
    return [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('j48_exact', DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, class_weight='balanced', random_state=42)),
        ('reptree_approx', DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42))
    ]

def build_stacking_model():
    """Builds the custom Stacking Classifier with a Logistic Regression meta-learner."""
    meta_learner = LogisticRegression(class_weight='balanced', random_state=42)
    return StackingClassifier(estimators=get_base_models(), final_estimator=meta_learner, cv=5)

def build_voting_model():
    """Builds a soft-voting classifier."""
    return VotingClassifier(estimators=get_base_models(), voting='soft')

def build_weka_mlp(n_features, n_classes=2):
    """
    Builds an MLP matching WEKA's default configuration.
    WEKA's 'a' = (features + classes) / 2.
    """
    # Calculate WEKA's 'a'
    hidden_nodes = int((n_features + n_classes) / 2)
    
    return MLPClassifier(
        hidden_layer_sizes=(hidden_nodes,),
        activation='logistic',       # WEKA uses sigmoid by default
        solver='sgd',                # Required to use momentum and learning rate
        learning_rate_init=0.3,      # User specified
        momentum=0.2,                # User specified
        alpha=0.01,
        max_iter=500,                # Training time
        random_state=42
    )

def build_lr_pipeline():
    """Builds a Logistic Regression pipeline scaling only numerical columns."""
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols)
        ], remainder='passthrough'
    )
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
    ])