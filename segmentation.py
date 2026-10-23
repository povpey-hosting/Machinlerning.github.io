"""ផ្នែកគណនា K-Means (Backend logic)"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "Mall_Customers.csv")


def load_data():
    """ផ្ទុក CSV ពិត ប្រសិនបើគ្មាន បង្កើតទិន្នន័យគំរូ"""
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH), True
    rng = np.random.default_rng(RANDOM_STATE)
    groups = [(26, 7, 20, 10, 40), (26, 7, 78, 10, 40), (55, 9, 50, 8, 60),
              (88, 10, 17, 9, 30), (88, 10, 82, 9, 30)]
    inc, sc = [], []
    for mi, si, ms, ss, n in groups:
        inc += list(rng.normal(mi, si, n))
        sc += list(rng.normal(ms, ss, n))
    n = len(inc)
    df = pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 70, n),
        FEATURES[0]: np.clip(inc, 15, 140).round().astype(int),
        FEATURES[1]: np.clip(sc, 1, 99).round().astype(int),
    })
    return df, False


DF, IS_REAL = load_data()
SCALER = StandardScaler()
X_SCALED = SCALER.fit_transform(DF[FEATURES].values)
INC_MID = DF[FEATURES[0]].median()
SPD_MID = DF[FEATURES[1]].median()


def k_metrics(k_min=2, k_max=10):
    ks, inertia, sil = [], [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_SCALED)
        ks.append(k)
        inertia.append(round(float(km.inertia_), 2))
        sil.append(round(float(silhouette_score(X_SCALED, labels)), 4))
    return {"k": ks, "inertia": inertia, "silhouette": sil,
            "best_k": ks[int(np.argmax(sil))]}


METRICS = k_metrics()


def name_segment(inc, spd):
    hi_inc, hi_spd = inc >= INC_MID, spd >= SPD_MID
    if abs(inc - INC_MID) < 12 and abs(spd - SPD_MID) < 12:
        return ("Standard (ទូទៅ)",
                "ផ្តល់ប្រូម៉ូសិនទូទៅ និង loyalty program ដើម្បីរក្សាអតិថិជន")
    if hi_inc and hi_spd:
        return ("VIP (ចំណូលខ្ពស់ ចំណាយច្រើន)",
                "ផ្តល់សេវាកម្ម VIP ផលិតផលពិសេស និងការអញ្ជើញព្រឹត្តិការណ៍")
    if hi_inc and not hi_spd:
        return ("Careful (ចំណូលខ្ពស់ ចំណាយតិច)",
                "ផ្សព្វផ្សាយផលិតផលគុណភាពខ្ពស់ និងការផ្តល់តម្លៃបន្ថែមផ្ទាល់ខ្លួន")
    if not hi_inc and hi_spd:
        return ("Impulsive (ចំណូលទាប ចំណាយច្រើន)",
                "ផ្តល់ការបញ្ចុះតម្លៃ installment និងប្រូម៉ូសិនរយៈពេលខ្លី")
    return ("Budget (ចំណូលទាប ចំណាយតិច)",
            "ផ្តោតលើតម្លៃសេដ្ឋកិច្ច និង coupon ដើម្បីទាក់ទាញ")


_model_cache = {}


def fit_model(k):
    if k not in _model_cache:
        _model_cache[k] = KMeans(n_clusters=k, n_init=10,
                                 random_state=RANDOM_STATE).fit(X_SCALED)
    return _model_cache[k]


def analyze(k):
    km = fit_model(k)
    labels = km.labels_
    df = DF.copy()
    df["Cluster"] = labels
    centroids = SCALER.inverse_transform(km.cluster_centers_)

    summary = []
    for c in range(k):
        sub = df[df["Cluster"] == c]
        inc, spd = sub[FEATURES[0]].mean(), sub[FEATURES[1]].mean()
        name, strategy = name_segment(inc, spd)
        summary.append({
            "cluster": c, "customers": int(len(sub)),
            "avg_age": round(float(sub["Age"].mean()), 1),
            "avg_income": round(float(inc), 1),
            "avg_spending": round(float(spd), 1),
            "female_pct": round(float((sub["Gender"] == "Female").mean() * 100), 1),
            "name": name, "strategy": strategy,
        })
    return {
        "k": k,
        "silhouette": round(float(silhouette_score(X_SCALED, labels)), 4),
        "is_real_data": IS_REAL,
        "total": int(len(df)),
        "points": [{"x": float(r[FEATURES[0]]), "y": float(r[FEATURES[1]]),
                    "c": int(r["Cluster"])} for _, r in df.iterrows()],
        "centroids": [{"x": round(float(a), 1), "y": round(float(b), 1), "c": i}
                      for i, (a, b) in enumerate(centroids)],
        "summary": summary,
        "metrics": METRICS,
    }


def predict(k, income, spending):
    km = fit_model(k)
    c = int(km.predict(SCALER.transform([[income, spending]]))[0])
    s = analyze(k)["summary"][c]
    return {"cluster": c, "name": s["name"], "strategy": s["strategy"]}
