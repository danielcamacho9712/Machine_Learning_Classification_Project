from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from src.encoding import fit_and_save_label_encoder


def evaluate_model(model, X_test, y_test, label_encoder=None):
    """ Evaluate the given model on the test data and return performance metrics and confusion matrix. """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = {
        'precision': precision_score(y_test, y_pred, average="weighted"),
        'recall': recall_score(y_test, y_pred, average="weighted"),
        'f1_score': f1_score(y_test, y_pred, average="weighted"),
        'roc_auc': roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="weighted"
        )
    }

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=label_encoder.classes_,
        cmap="Blues"
    )

    disp.ax_.set_title(
        f'Confusion Matrix of {model.named_steps["classifier"].__class__.__name__}'
    )

    return results, disp


def evaluate_train_data(model, X_train, y_train):

    """ Evaluate the given model on the training data and return performance metrics, to rule out overfitting. """

    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)

    results = {
        'precision': precision_score(y_train, y_pred, average="weighted"),
        'recall': recall_score(y_train, y_pred, average="weighted"),
        'f1_score': f1_score(y_train, y_pred, average="weighted"),
        'roc_auc': roc_auc_score(
            y_train,
            y_proba,
            multi_class="ovr",
            average="weighted"
        )
    }

    return results