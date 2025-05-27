from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def one_hot_encoder(df, nan_as_category=True):
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    return df, categorical_columns

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        # Nettoyage spécifique
        if 'CODE_GENDER' in df.columns:
            df = df[df['CODE_GENDER'] != 'XNA']

        # Encodage binaire
        for col in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            if col in df.columns:
                df[col], _ = pd.factorize(df[col])

        # One-hot encoding
        df, _ = one_hot_encoder(df)

        # Remplacements spécifiques
        if 'DAYS_EMPLOYED' in df.columns:
            df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
            if 'DAYS_BIRTH' in df.columns:
                df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

        # Feature engineering métiers
        if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
            df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

        if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
            df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

        if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
            df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

        return df




