from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from src.preprocessing import target_encoder


def evaluate_model(model, test_data):

    X_test = test_data.drop(columns=['Weather Type'])
    y_test = test_data['Weather Type']

    y_test_encoded = target_encoder.transform(y_test)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = {
        'precision': precision_score(y_test_encoded, y_pred, average="weighted"),
        'recall': recall_score(y_test_encoded, y_pred, average="weighted"),
        'f1_score': f1_score(y_test_encoded, y_pred, average="weighted"),
        'roc_auc': roc_auc_score(
            y_test_encoded,
            y_proba,
            multi_class="ovr",
            average="weighted"
        )
    }

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test_encoded,
        y_pred,
        display_labels=target_encoder.classes_,
        cmap="Blues"
    )

    disp.ax_.set_title(
        f'Confusion Matrix of {model.named_steps["classifier"].__class__.__name__}'
    )

    return results, disp


def evaluate_train_data(model, train_data):

    X_train = train_data.drop(columns=['Weather Type'])
    y_train = train_data['Weather Type']

    y_train_encoded = target_encoder.transform(y_train)

    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)

    results = {
        'precision': precision_score(y_train_encoded, y_pred, average="weighted"),
        'recall': recall_score(y_train_encoded, y_pred, average="weighted"),
        'f1_score': f1_score(y_train_encoded, y_pred, average="weighted"),
        'roc_auc': roc_auc_score(
            y_train_encoded,
            y_proba,
            multi_class="ovr",
            average="weighted"
        )
    }

    return results