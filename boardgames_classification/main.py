import sys
import json
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def is_spam(t):
    raw = re.sub(r'<[^>]+>', ' ', str(t)).strip()
    if len(raw) < 5: return True
    ns = [c for c in raw if not c.isspace()]
    if not ns or sum(1 for c in ns if c.isalpha()) / len(ns) < 0.30: return True
    words = raw.lower().split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.15: return True
    return False

def std_clean(t):
    t = re.sub(r'<[^>]+>', ' ', str(t))
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'[^\w\s]', ' ', t, flags=re.UNICODE)
    t = re.sub(r'\b\d+\b', ' ', t)
    return ' '.join(t.lower().split())

def raw_clean(t):
    return re.sub(r'<[^>]+>', ' ', str(t)).lower().strip()

def extract_year_features(df):
    yr = pd.to_datetime(df['creation_date'], errors='coerce').dt.year.fillna(2015).values
    return np.column_stack([yr, np.log1p(np.clip(yr - 2010, 0, 20))])

def main():
    if len(sys.argv) < 3:
        return

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    with open(train_path, 'r', encoding='utf-8') as f:
        df_train = pd.DataFrame(json.load(f))
        
    with open(test_path, 'r', encoding='utf-8') as f:
        df_test = pd.DataFrame(json.load(f))

    df_train = df_train.dropna(subset=['label']).copy()
    df_train['text'] = df_train['text'].fillna('')
    
    for col in ['score', 'views', 'answers', 'comments', 'favorites']:
        mask = pd.to_numeric(df_train[col], errors='coerce') < -10
        df_train = df_train[~mask.fillna(False)]
    
    df_train = df_train[~df_train['text'].apply(is_spam)].copy()

    X_num_iso = pd.DataFrame({col: np.log1p(pd.to_numeric(df_train[col], errors='coerce').fillna(0).clip(lower=0))
                              for col in ['score', 'views', 'answers', 'comments', 'favorites']})
    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    df_train = df_train[iso.fit_predict(X_num_iso) == 1].copy().reset_index(drop=True)

    y_train = df_train['label']

    best_params = {'wn': 0.32, 'ws': 0.258, 'wr': 0.422, 'C_num': 5}

    X_tr_std = df_train['text'].apply(std_clean).values
    X_tr_raw = df_train['text'].apply(raw_clean).values
    
    c1s = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=3, max_df=0.92, sublinear_tf=True, max_features=25000)
    c2s = TfidfVectorizer(analyzer='char_wb', ngram_range=(4,6), min_df=3, max_df=0.92, sublinear_tf=True, max_features=18000)
    c1r = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,7), min_df=2, max_df=0.92, sublinear_tf=True, max_features=80000)
    
    Xm_std_tr = sp.hstack([c1s.fit_transform(X_tr_std), c2s.fit_transform(X_tr_std)*0.7], format='csr')
    Xm_raw_tr = c1r.fit_transform(X_tr_raw)
    
    num_yr_tr = extract_year_features(df_train)
    sc = StandardScaler()
    ny_tr_s = sc.fit_transform(num_yr_tr)

    lr_std = LogisticRegression(C=10, max_iter=2000, class_weight='balanced', random_state=42)
    lr_std.fit(Xm_std_tr, y_train)

    lr_raw = LogisticRegression(C=10, max_iter=2000, class_weight='balanced', random_state=42)
    lr_raw.fit(Xm_raw_tr, y_train)

    lr_num = LogisticRegression(C=best_params['C_num'], max_iter=1000, class_weight='balanced', random_state=42)
    lr_num.fit(ny_tr_s, y_train)
    
    classes = lr_std.classes_

    X_te_std = df_test['text'].fillna('').apply(std_clean).values
    X_te_raw = df_test['text'].fillna('').apply(raw_clean).values
    
    Xm_std_te = sp.hstack([c1s.transform(X_te_std), c2s.transform(X_te_std)*0.7], format='csr')
    Xm_raw_te = c1r.transform(X_te_raw)
    ny_te_s = sc.transform(extract_year_features(df_test))

    p_std = lr_std.predict_proba(Xm_std_te)
    p_raw = lr_raw.predict_proba(Xm_raw_te)
    p_num = lr_num.predict_proba(ny_te_s)

    p_final = (p_std * best_params['ws']) + (p_raw * best_params['wr']) + (p_num * best_params['wn'])
    y_pred = classes[np.argmax(p_final, axis=1)]

    if 'label' in df_test.columns:
        print(f1_score(df_test['label'], y_pred, average='macro'))

if __name__ == "__main__":
    main()