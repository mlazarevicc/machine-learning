import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CLUSTER_FEATURE = "log_base_exp"
KM_K = 2
KM_N_INIT = 300
KM_MAX_ITER = 2000
KM_TOL = 1e-6

def load(path):
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    return df

def engineer(df):
    df = df.copy()
    df[CLUSTER_FEATURE] = np.log1p(df["base_experience"])
    return df

def main():
    train_path = sys.argv[1]
    test_path  = sys.argv[2]

    train_df = engineer(load(train_path))
    test_df  = engineer(load(test_path))

    ss = StandardScaler()
    X_train = ss.fit_transform(train_df[[CLUSTER_FEATURE]].values)
    X_test  = ss.transform(test_df[[CLUSTER_FEATURE]].values)

    km = KMeans(
        n_clusters=KM_K,
        algorithm="lloyd",
        init="k-means++",
        n_init=KM_N_INIT,
        max_iter=KM_MAX_ITER,
        tol=KM_TOL,
        random_state=RANDOM_STATE,
    )
    km.fit(X_train)
    labels_test = km.predict(X_test)

    score = silhouette_score(X_test, labels_test)
    print(score)

if __name__ == "__main__":
    main()