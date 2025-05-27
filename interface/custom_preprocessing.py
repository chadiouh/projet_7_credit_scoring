from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_features = [
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
            "PAYMENT_RATE", "DAYS_EMPLOYED", "AMT_GOODS_PRICE",
            "OWN_CAR_AGE", "AMT_ANNUITY"
        ]
        self.categorical_features = [
            "CODE_GENDER", "NAME_EDUCATION_TYPE_Higher_education"
        ]

    def fit(self, X, y=None):
        self.pipeline = ColumnTransformer([
            ("num", StandardScaler(), self.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features)
        ])
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


