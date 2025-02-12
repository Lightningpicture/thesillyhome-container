import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, auc
from sklearn.preprocessing import MinMaxScaler

import thesillyhome.model_creator.read_config_json as tsh_config

def save_visual_tree(model, actuator, feature_vector):
    """Saves a visual representation of a decision tree."""
    plt.figure(figsize=(12, 12))
    plot_tree(model, fontsize=10, feature_names=feature_vector, max_depth=7)
    plt.savefig(f"/thesillyhome_src/frontend/static/data/{actuator}_tree.png")
    plt.close()

def to_labels(pos_probs, threshold):
    """Converts probabilities to binary labels based on a threshold."""
    return (pos_probs >= threshold).astype("int")

def optimization_function(precision, recall):
    """Calculates the optimal threshold based on a custom optimization metric."""
    epsilon = 0.01
    optimizer = (2 * precision * recall) / (1 / 5 * precision + recall + epsilon)
    ix = np.argmax(optimizer)
    return ix, optimizer

def train_all_actuator_models():
    """Trains models for each actuator."""
    actuators = tsh_config.actuators
    df_act_states = pd.read_pickle(f"{tsh_config.data_dir}/parsed/act_states.pkl").reset_index(drop=True)

    output_list = tsh_config.output_list.copy()
    act_list = list(set(df_act_states.columns) - set(output_list))

    model_types = {
        "DecisionTreeClassifier": {
            "classifier": DecisionTreeClassifier,
            "model_kwargs": {
                "max_depth": 7,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
            },
        },
        "LogisticRegression": {
            "classifier": LogisticRegression,
            "model_kwargs": {
                "max_iter": 10000,
                "C": 0.5,
            },
        },
        "RandomForestClassifier": {
            "classifier": RandomForestClassifier,
            "model_kwargs": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
            },
        },
        "SVMClassifier": {
            "classifier": SVC,
            "model_kwargs": {
                "probability": True,
                "C": 0.5,
            },
        },
    }
    
    metrics_matrix = []

    for actuator in actuators:
        logging.info(f"Training model for {actuator}")
        df_act = df_act_states[df_act_states["entity_id"] == actuator]

        if df_act.empty:
            logging.info(f"No cases found for {actuator}")
            continue

        if len(df_act) < 30:
            logging.info("Samples less than 30. Skipping")
            continue

        if df_act["state"].nunique() == 1:
            logging.info(f"All cases for {actuator} have the same state. Skipping")
            continue

        output_vector = df_act["state"]
        cur_act_list = [feature for feature in act_list if feature.startswith(actuator)]
        feature_list = sorted(list(set(act_list) - set(cur_act_list)))
        feature_vector = df_act[feature_list]

        X_train, X_test, y_train, y_test = train_test_split(feature_vector, output_vector, test_size=0.1)

        base_weight = 0.4
        n_samples = len(X_train)
        recent_weight = np.logspace(0.4, 0.2, n_samples, base=2)
        scaler = MinMaxScaler(feature_range=(base_weight, 0.6))
        sample_weight = scaler.fit_transform(recent_weight.reshape(-1, 1)).flatten()

        if "duplicate" in X_train.columns:
            sample_weight *= X_train["duplicate"]
            X_train = X_train.drop(columns="duplicate")
            X_test = X_test.drop(columns="duplicate")

        train_all_classifiers(
            model_types,
            actuator,
            X_train,
            X_test,
            y_train,
            y_test,
            sample_weight,
            metrics_matrix,
            feature_list,
        )

    save_metrics(metrics_matrix)
    logging.info("Completed!")

def save_metrics(metrics_matrix):
    """Saves the metrics to a file."""
    df_metrics_matrix = pd.DataFrame(metrics_matrix)

    metrics_path = "/thesillyhome_src/frontend/static/data/metrics_matrix.json"

    # Falls keine Daten vorhanden sind, erstelle eine leere JSON-Datei
    if df_metrics_matrix.empty:
        logging.warning("No data available for metrics. Creating an empty metrics_matrix.json file.")
        with open(metrics_path, "w") as f:
            f.write("[]")
        return  # Verhindert Fehler durch leere Daten

    df_metrics_matrix.to_pickle(f"/thesillyhome_src/data/model/metrics.pkl")

    try:
        best_metrics_matrix = df_metrics_matrix.fillna(0).sort_values(
            "best_optimizer", ascending=False
        ).drop_duplicates(subset=["actuator"], keep="first")
    except Exception as e:
        logging.warning(f"No metrics available: {e}")
        with open(metrics_path, "w") as f:
            f.write("[]")  # Sicherheitshalber leere JSON-Datei erstellen
        return

    best_metrics_matrix.to_json(metrics_path, orient="records")

def save_model(model, filepath):
    """Saves a model to a specified filepath."""
    with open(filepath, "wb") as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        filename="/thesillyhome_src/log/thesillyhome.log",
        encoding="utf-8",
        level=logging.INFO,
        format=FORMAT,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(FORMAT))
    logging.getLogger().addHandler(handler)

    train_all_actuator_models()
