import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

import thesillyhome.model_creator.read_config_json as tsh_config

def save_visual_tree(model, actuator, feature_vector):
    plt.figure(figsize=(12, 12))
    plot_tree(model, fontsize=10, feature_names=feature_vector, max_depth=7)
    plt.savefig(f"/thesillyhome_src/frontend/static/data/{actuator}_tree.png")
    plt.close()

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype("int")

def optimization_function(precision, recall):
    epsilon = 0.01
    optimizer = (2 * precision * recall) / (1 / 5 * precision + recall + epsilon)
    ix = np.argmax(optimizer)
    return ix, optimizer

def train_all_actuator_models():
    actuators = tsh_config.actuators
    df_act_states = pd.read_pickle(f"{tsh_config.data_dir}/parsed/act_states.pkl").reset_index(drop=True)

    output_list = tsh_config.output_list.copy()
    act_list = list(set(df_act_states.columns) - set(output_list))

    model_types = {
        "DecisionTreeClassifier": {"classifier": DecisionTreeClassifier, "model_kwargs": {"max_depth": 5}},
        "LogisticRegression": {"classifier": LogisticRegression, "model_kwargs": {"max_iter": 5000, "C": 1.0}},
        "RandomForestClassifier": {"classifier": RandomForestClassifier, "model_kwargs": {"n_estimators": 50}},
        "SVMClassifier": {"classifier": SVC, "model_kwargs": {"probability": True, "kernel": "linear", "C": 0.5}},
    }

    metrics_matrix = []

    Parallel(n_jobs=-1)(delayed(train_actuator_model)(
        actuator, df_act_states, model_types, act_list, metrics_matrix
    ) for actuator in actuators)

    save_metrics(metrics_matrix)
    logging.info("Completed!")

def train_actuator_model(actuator, df_act_states, model_types, act_list, metrics_matrix):
    logging.info(f"Training model for {actuator}")
    df_act = df_act_states[df_act_states["entity_id"] == actuator]

    if df_act.empty or len(df_act) < 30 or df_act["state"].nunique() == 1:
        logging.info(f"Skipping {actuator} due to insufficient data.")
        return

    output_vector = df_act["state"]
    feature_list = sorted(list(set(act_list) - set([feature for feature in act_list if feature.startswith(actuator)])))
    feature_vector = df_act[feature_list]

    # Stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(feature_vector, output_vector, test_size=0.3, stratify=output_vector)

    # Feature selection
    selector = SelectKBest(f_classif, k=min(20, len(feature_vector.columns)))
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # Sample weight calculation
    n_samples = len(X_train)
    sample_weight = np.logspace(0.1, 0.6, n_samples, base=2)
    scaler = MinMaxScaler(feature_range=(0.4, 0.7))
    sample_weight = scaler.fit_transform(sample_weight.reshape(-1, 1)).flatten()

    train_all_classifiers(model_types, actuator, X_train, X_test, y_train, y_test, sample_weight, metrics_matrix, feature_list)

def train_all_classifiers(model_types, actuator, X_train, X_test, y_train, y_test, sample_weight, metrics_matrix, feature_list):
    logging.info(f"---Training samples = {len(y_train)}")

    model_directory = f"{tsh_config.data_dir}/model/{actuator}"
    os.makedirs(model_directory, exist_ok=True)

    best_model = 0
    for model_name, model_vars in model_types.items():
        logging.info(f"---Running training for {model_name}")

        model = model_vars["classifier"](**model_vars["model_kwargs"])
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except Exception as e:
            logging.warning(f"Training failed for {model_name} on {actuator}: {e}")
            continue

        if model_name == "DecisionTreeClassifier":
            save_visual_tree(model, actuator, feature_list)

        if len(model.classes_) > 1:
            y_predictions_proba = model.predict_proba(X_test)[:, 1]
        else:
            logging.warning(f"Skipping {actuator} with {model_name}: only one class present.")
            continue

        precision, recall, thresholds = precision_recall_curve(y_test, y_predictions_proba)
        ix, optimizer = optimization_function(precision, recall)
        auc_ = auc(recall, precision)

        plt.plot(precision, recall, label=model_name)
        plt.scatter(precision[ix], recall[ix], marker="o", label=f"{model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        y_predictions_best = to_labels(y_predictions_proba, thresholds[ix])

        metrics_json = {
            "actuator": actuator,
            "classifier_name": model_name,
            "accuracy": accuracy_score(y_test, y_predictions_best),
            "precision": precision_score(y_test, y_predictions_best),
            "recall": recall_score(y_test, y_predictions_best),
            "AUC": auc_,
            "best_thresh": thresholds[ix],
            "best_optimizer": optimizer[ix],
            "model_enabled": False,
        }

        metrics_matrix.append(metrics_json)

        if optimizer[ix] > best_model and metrics_json["precision"] > 0.7:
            metrics_json["model_enabled"] = True
            best_model = optimizer[ix]
            save_model(model, f"{model_directory}/best_model.pkl")

        save_model(model, f"{model_directory}/{model_name}.pkl")

    save_plots(actuator, y_train)

def save_model(model, filepath):
    with open(filepath, "wb") as file:
        pickle.dump(model, file)

def save_plots(actuator, y_train):
    plt.plot([0, 1], [y_train.sum() / len(y_train), y_train.sum() / len(y_train)], linestyle="--", label="No Skill")
    plt.savefig(f"/thesillyhome_src/frontend/static/data/{actuator}_precision_recall.png")
    plt.close()

def save_metrics(metrics_matrix):
    df_metrics_matrix = pd.DataFrame(metrics_matrix)
    df_metrics_matrix.to_pickle(f"/thesillyhome_src/data/model/metrics.pkl")
    try:
        best_metrics_matrix = df_metrics_matrix.fillna(0).sort_values("best_optimizer", ascending=False).drop_duplicates(subset=["actuator"], keep="first")
    except Exception as e:
        logging.warning(f"No metrics available: {e}")
        return
    metrics_path = "/thesillyhome_src/frontend/static/data/metrics_matrix.json"
    best_metrics_matrix.to_json(metrics_path, orient="records")

if __name__ == "__main__":
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename="/thesillyhome_src/log/thesillyhome.log", encoding="utf-8", level=logging.INFO, format=FORMAT)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(FORMAT))
    logging.getLogger().addHandler(handler)

    train_all_actuator_models()
