import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


def to_num(series, fill=None):
    s = pd.to_numeric(series, errors='coerce')
    return s.fillna(fill) if fill is not None else s

def map_floor(f):
    f = str(f).lower().strip()
    if any(x in f for x in ['prizemlje', 'vpr', '0', 'nisko prizemlje']):
        return 0.0
    if any(x in f for x in ['suteren', '-1']):
        return -1.0
    if any(x in f for x in ['potkrovlje', 'pk']):
        return 10.0
    try:
        v = float(f)
        return v if v <= 30 else 3.0
    except:
        return 1.0

def preprocess(df, medians):
    d = df.copy()

    d['Kvadratura'] = to_num(d['Kvadratura'], medians['area'])
    d['Sobe'] = to_num(d['Sobe'], medians['rooms'])
    d['Slike'] = to_num(d['Slike'], medians['images'])

    d['Log_Kvadratura'] = np.log1p(d['Kvadratura'])
    d['Area_per_Room'] = d['Kvadratura'] / (d['Sobe'] + 0.1)

    for col in ['Uknjizen', 'Garaza', 'Parking']:
        if col in d:
            d[col] = d[col].map({'Da': 1, 'Ne': 0}).fillna(0)

    if {'Garaza', 'Parking'}.issubset(d.columns):
        d['Ima_Parking_Ili_Garazu'] = ((d['Garaza'] == 1) | (d['Parking'] == 1)).astype(int)

    n = d['Naziv'].str.lower().fillna('')
    d['Lux'] = n.str.contains('lux|luks|novo|premium|hram|bw|vodi|dedinj|senjak|salonac|smart').astype(int)
    d['Renoviranje'] = n.str.contains('izvorno|renovir|hitno|suteren|ulaganj|stara|adaptacija').astype(int)

    d['Sprat_Num'] = d['Sprat'].apply(map_floor).fillna(medians['floor'])
    d['Je_Prizemlje'] = (d['Sprat_Num'] == 0).astype(int)
    d['Je_Suteren'] = (d['Sprat_Num'] == -1).astype(int)

    if 'Prodavac' in d:
        d['Je_Investitor'] = (d['Prodavac'].str.lower() == 'investitor').astype(int)

    d['Lokacija_Spojeno'] = d['Grad'].astype(str) + "_" + d['Naziv'].astype(str)

    d = d.drop(columns=['Naziv', 'Sprat', 'Prodavac', 'm2_temp'], errors='ignore')
    d = pd.get_dummies(d, columns=['Grad'], drop_first=True)

    for col in d.columns:
        if col not in ['Cena', 'Lokacija_Spojeno']:
            d[col] = to_num(d[col], 0)

    return d

def main():
    if len(sys.argv) != 3:
        return

    train = pd.read_json(sys.argv[1]).drop_duplicates()
    test  = pd.read_json(sys.argv[2])

    train['Kvadratura'] = to_num(train['Kvadratura'])
    train['Sobe'] = to_num(train['Sobe'])
    train = train.dropna(subset=['Cena', 'Kvadratura', 'Sobe'])

    train['m2_temp'] = train['Cena'] / train['Kvadratura']
    train = train[
        (train['Cena'] > 15000) &
        (train['m2_temp'].between(500, 8000)) &
        (np.abs(stats.zscore(np.log1p(train['Cena']))) < 3)
    ]

    medians = {
        'images': to_num(train['Slike']).median(),
        'area': train['Kvadratura'].median(),
        'rooms': train['Sobe'].median(),
        'floor': train['Sprat'].apply(map_floor).median()
    }

    train_p = preprocess(train, medians)
    test_p = preprocess(test, medians)

    X_train = train_p.drop(columns=['Cena'])
    y_train = train_p['Cena']

    X_test = test_p.drop(columns=['Cena']).reindex(columns=X_train.columns, fill_value=0)
    X_test['Lokacija_Spojeno'] = test_p.get('Lokacija_Spojeno', "Nepoznato")

    y_test = test_p.get('Cena')

    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ('prep', ColumnTransformer([
                ('target_enc', TargetEncoder(smooth='auto'), ['Lokacija_Spojeno'])
            ], remainder='passthrough')),
            ('scaler', RobustScaler()),
            ('ridge', Ridge(alpha=1.0))
        ]),
        func=np.log1p, inverse_func=np.expm1
    )

    model.fit(X_train, y_train)

    if y_test is not None:
        preds = model.predict(X_test)
        print(mean_absolute_percentage_error(y_test, preds))

if __name__ == "__main__":
    main()