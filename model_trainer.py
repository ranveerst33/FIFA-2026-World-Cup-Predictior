"""
FIFA 2026 World Cup Prediction - ML Model Training & Evaluation
================================================================
Trains multiple ML models, evaluates them, builds a Voting Ensemble,
and saves everything to disk for prediction use.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available, skipping.")

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ModelTrainer:
    """Train, evaluate, and save FIFA World Cup prediction models."""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_cols = None
        self.scaler = None
        os.makedirs(MODELS_DIR, exist_ok=True)

    def prepare_data(self, X, y, feature_cols):
        """Split and scale the data."""
        self.feature_cols = feature_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=feature_cols,
            index=X_test.index,
        )

        self.X_train = X_train
        self.X_test = X_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"Train distribution: {dict(y_train.value_counts().sort_index())}")
        print(f"Test distribution:  {dict(y_test.value_counts().sort_index())}")
        return self

    def _build_models(self):
        """Define all models."""
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
            "Logistic Regression": LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        }

        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
                use_label_encoder=False,
            )

        return models

    def train_all(self):
        """Train all individual models + voting ensemble."""
        print("\n" + "=" * 60)
        print("TRAINING ML MODELS")
        print("=" * 60)

        base_models = self._build_models()

        for name, model in base_models.items():
            print(f"\n--- Training {name} ---")
            # Use scaled data for Logistic Regression, raw for tree-based
            if "Logistic" in name:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_proba = model.predict_proba(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)

            self.models[name] = model
            self._evaluate(name, y_pred, y_proba)

        # ----- Voting Ensemble -----
        print(f"\n--- Training Voting Ensemble ---")
        estimators = []
        for name, model in base_models.items():
            clean_name = name.lower().replace(" ", "_")
            estimators.append((clean_name, model))

        ensemble = VotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=-1,
        )
        ensemble.fit(self.X_train, self.y_train)
        y_pred_ens = ensemble.predict(self.X_test)
        y_proba_ens = ensemble.predict_proba(self.X_test)

        self.models["Voting Ensemble"] = ensemble
        self._evaluate("Voting Ensemble", y_pred_ens, y_proba_ens)

        return self

    def _evaluate(self, name, y_pred, y_proba):
        """Compute all evaluation metrics for a model."""
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred)

        self.results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "predictions": y_pred,
            "probabilities": y_proba,
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    def cross_validate(self):
        """Run 5-fold cross-validation on all models."""
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION (5-Fold)")
        print("=" * 60)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            if name == "Voting Ensemble":
                X_cv = self.X_train
            elif "Logistic" in name:
                X_cv = self.X_train_scaled
            else:
                X_cv = self.X_train

            try:
                scores = cross_val_score(model, X_cv, self.y_train, cv=cv, scoring="accuracy", n_jobs=-1)
                self.results[name]["cv_mean"] = scores.mean()
                self.results[name]["cv_std"] = scores.std()
                print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                print(f"  {name}: CV failed - {e}")

    def compute_feature_importance(self):
        """Extract feature importances from tree-based models."""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE")
        print("=" * 60)

        self.feature_importances = {}

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=self.feature_cols)
                imp = imp.sort_values(ascending=False)
                self.feature_importances[name] = imp
                print(f"\n  {name} - Top 10 features:")
                for feat, val in imp.head(10).items():
                    print(f"    {feat}: {val:.4f}")

        return self

    def save_models(self):
        """Save all trained models and metadata."""
        print("\nSaving models...")

        for name, model in self.models.items():
            clean_name = name.lower().replace(" ", "_")
            filepath = os.path.join(MODELS_DIR, f"{clean_name}.pkl")
            joblib.dump(model, filepath)
            print(f"  Saved: {filepath}")

        # Save scaler
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        print(f"  Saved: scaler.pkl")

        # Save feature columns
        with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
            json.dump(self.feature_cols, f)
        print(f"  Saved: feature_cols.json")

        # Save evaluation results (without numpy arrays)
        results_clean = {}
        for name, metrics in self.results.items():
            results_clean[name] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "confusion_matrix": metrics["confusion_matrix"],
                "cv_mean": metrics.get("cv_mean", None),
                "cv_std": metrics.get("cv_std", None),
            }

        with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "w") as f:
            json.dump(results_clean, f, indent=2)
        print(f"  Saved: evaluation_results.json")

        # Save feature importances
        if hasattr(self, "feature_importances"):
            fi_dict = {}
            for name, imp in self.feature_importances.items():
                fi_dict[name] = imp.to_dict()
            with open(os.path.join(MODELS_DIR, "feature_importances.json"), "w") as f:
                json.dump(fi_dict, f, indent=2)
            print(f"  Saved: feature_importances.json")

        print("[OK] All models and metadata saved to models/")
        return self

    def generate_report_plots(self):
        """Generate evaluation plots and save to models/ directory."""
        print("\nGenerating evaluation plots...")

        # ----- 1. Model comparison bar chart -----
        fig, ax = plt.subplots(figsize=(12, 6))
        model_names = list(self.results.keys())
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        x = np.arange(len(model_names))
        width = 0.2

        for i, metric in enumerate(metrics):
            vals = [self.results[m][metric] for m in model_names]
            bars = ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title())
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "model_comparison.png"), dpi=150)
        plt.close()
        print("  Saved: model_comparison.png")

        # ----- 2. Confusion matrices -----
        labels = ["Home Win", "Draw", "Away Win"]
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for ax, (name, metrics) in zip(axes, self.results.items()):
            cm = np.array(metrics["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=labels, yticklabels=labels)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.suptitle("Confusion Matrices", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "confusion_matrices.png"), dpi=150)
        plt.close()
        print("  Saved: confusion_matrices.png")

        # ----- 3. Feature importance (top model) -----
        if hasattr(self, "feature_importances") and self.feature_importances:
            best_model = max(self.feature_importances.keys(),
                            key=lambda k: self.results.get(k, {}).get("f1_score", 0))
            imp = self.feature_importances[best_model]

            fig, ax = plt.subplots(figsize=(10, 8))
            imp.sort_values().plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(f"Feature Importance ({best_model})")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(MODELS_DIR, "feature_importance.png"), dpi=150)
            plt.close()
            print("  Saved: feature_importance.png")

        print("[OK] All plots saved to models/")


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    from data_processor import DataProcessor

    print("=" * 60)
    print("   FIFA 2026 WORLD CUP PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Step 1: Build training data
    dp = DataProcessor()
    X, y, feature_cols = dp.build_training_data()

    # Step 2: Train models
    trainer = ModelTrainer()
    trainer.prepare_data(X, y, feature_cols)
    trainer.train_all()

    # Step 3: Cross-validate
    trainer.cross_validate()

    # Step 4: Feature importance
    trainer.compute_feature_importance()

    # Step 5: Save everything
    trainer.save_models()
    trainer.generate_report_plots()

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("   FINAL RESULTS SUMMARY")
    print("=" * 60)
    for name, metrics in trainer.results.items():
        cv = f"CV={metrics.get('cv_mean', 0):.4f}(±{metrics.get('cv_std', 0):.4f})" if metrics.get('cv_mean') else ""
        print(f"  {name:25s} | Acc={metrics['accuracy']:.4f} | F1={metrics['f1_score']:.4f} | {cv}")

    print("\n[OK] Training complete! Models saved to models/")
