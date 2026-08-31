# =============================================================================
#  train_demand_model.py
# train a simple logistic regression model to predict booking probability
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

CSV_PATH = "hotel_bookings.csv"    
ROOM_TYPE = "A"                     

# ====================== Load csv data ==================================
def load_and_filter(csv_path=CSV_PATH, room_type=ROOM_TYPE):
    print("=" * 58)
    print("  Step 1 — Load and filter Kaggle data")
    print("=" * 58)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} total bookings")

    # only one room type
    df = df[df["reserved_room_type"] == room_type].copy()
    print(f"  Filtered to room type '{room_type}': {len(df):,} bookings")

    # drop rows with missing essential fields
    df = df.dropna(subset=["adr", "lead_time", "arrival_date_month"])
    print(f"  After dropping missing values: {len(df):,} bookings\n")

    return df

# =======================Feature engineering=================================
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

def engineer_features(df):
    print("=" * 58)
    print("  Step 2 — Feature engineering")
    print("=" * 58)

    # ── Price ──────────────────────────────────────────────────────────────
    # adr = Average Daily Rate — this IS the price feature.
    # Remove obvious data errors (adr <= 0 or extreme outliers)
    df = df[(df["adr"] > 0) & (df["adr"] < 1000)].copy()

    # ── Month (numeric, then one-hot encoded) ─────────────────────────────
    # One-hot encoding avoids the "December is far from January" problem
    # that plain numeric encoding would create in a linear model
    df["month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    month_dummies = pd.get_dummies(df["month_num"], prefix="month")

    # ── Is weekend ─────────────────────────────────────────────────────────
    # Reconstruct the actual arrival date and get real day of week
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["month_num"].astype(str) + "-" +
        df["arrival_date_day_of_month"].astype(str),
        errors="coerce"
    )
    df = df.dropna(subset=["arrival_date"])
    df["is_weekend"] = (df["arrival_date"].dt.dayofweek >= 5).astype(int)

    # ── Target: booked = 1 if NOT cancelled, 0 if cancelled ────────────────
    df["booked"] = (df["is_canceled"] == 0).astype(int)

    # ── Assemble final feature table ────────────────────────────────────────
    # Reset df's index too, so it stays aligned with features/target by
    # position — this is what chronological_split() relies on.
    df = df.reset_index(drop=True)
    features = pd.concat([
        df[["adr", "lead_time", "is_weekend"]],
        month_dummies.reset_index(drop=True),
    ], axis=1)
    target = df["booked"]

    print(f"  Features used : adr (price), lead_time, is_weekend, "
          f"12 month dummies")
    print(f"  Total rows    : {len(features):,}")
    print(f"  Booked (1)    : {(target==1).sum():,}  "
          f"({(target==1).mean()*100:.1f}%)")
    print(f"  Cancelled (0) : {(target==0).sum():,}  "
          f"({(target==0).mean()*100:.1f}%)\n")

    return df, features, target


# ====================TRAIN / TEST SPLIT (chronological, not random)=========================

def chronological_split(df, features, target, train_frac=0.70):
    print("=" * 58)
    print("  Step 3 — Chronological train/test split")
    print("=" * 58)

    # Sort by arrival date so the split respects real time order.
    # df, features and target all share the same positional index
    # (guaranteed by the reset_index(drop=True) calls in engineer_features).
    order = df["arrival_date"].sort_values().index
    split_idx = int(len(order) * train_frac)

    train_idx = order[:split_idx]
    test_idx  = order[split_idx:]

    X_train, X_test = features.loc[train_idx], features.loc[test_idx]
    y_train, y_test = target.loc[train_idx],   target.loc[test_idx]

    print(f"  Train set: {len(X_train):,} bookings  ({train_frac*100:.0f}%)")
    print(f"  Test set : {len(X_test):,} bookings  ({(1-train_frac)*100:.0f}%)\n")

    return X_train, X_test, y_train, y_test


# ===============Train Logistic Regression Model=========================

def train_model(X_train, y_train, X_test, y_test):
    print("=" * 58)
    print("  Step 4 — Train Logistic Regression")
    print("=" * 58)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Cancelled", "Booked"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"  (0.5 = random guessing, 1.0 = perfect prediction)\n")

    return model


# ==============Validate model behavior=========================

def validate_model(model, feature_columns, save_path="demand_validation.png"):
    """
    Sanity checks: does the model predict lower booking probability
    at higher prices? This MUST be true or the demand model is broken.
    """
    print("=" * 58)
    print("  Step 5 — Validate demand model behaviour")
    print("=" * 58)

    # Build a synthetic sweep: fixed month=August, weekday, vary price
    prices = np.linspace(30, 300, 50)
    month_cols = [c for c in feature_columns if c.startswith("month_")]

    rows = []
    for price in prices:
        row = {c: 0 for c in feature_columns}
        row["adr"]        = price
        row["lead_time"]  = 30      # fixed for this sweep
        row["is_weekend"] = 0
        # August = month 8
        aug_col = "month_8"
        if aug_col in row:
            row[aug_col] = 1
        rows.append(row)

    sweep_df = pd.DataFrame(rows)[feature_columns]
    probs = model.predict_proba(sweep_df)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prices, probs, color="#2E75B6", linewidth=2)
    ax.set_xlabel("Price (€)", fontsize=12)
    ax.set_ylabel("Predicted Booking Probability", fontsize=12)
    ax.set_title("Demand Model Validation\n(August, weekday, 30-day lead time)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f" ! Saved validation plot: {save_path}")

    is_decreasing = probs[-1] < probs[0]
    print(f"  Price €30  → P(booking) = {probs[0]:.3f}")
    print(f"  Price €300 → P(booking) = {probs[-1]:.3f}")
    if is_decreasing:
        print(f" ! PASS — probability decreases as price increases")
    else:
        print(f" !!! FAIL — probability does not decrease with price. "
              f"Check feature engineering.")
    print()


# ==============Save model =========================

def save_model(model, feature_columns):
    print("=" * 58)
    print("  Step 6 — Save model")
    print("=" * 58)

    joblib.dump(model, "demand_model.pkl")
    joblib.dump(feature_columns, "demand_features.pkl")

    print(f" ! Saved: demand_model.pkl")
    print(f" ! Saved: demand_features.pkl")
    print(f"     ({len(feature_columns)} features, in order)\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = load_and_filter()
    df, features, target = engineer_features(df)
    X_train, X_test, y_train, y_test = chronological_split(df, features, target)
    model = train_model(X_train, y_train, X_test, y_test)
    validate_model(model, list(features.columns))
    save_model(model, list(features.columns))

    print("=" * 58)
    print("  Done. Run DL train")
    print("=" * 58)
