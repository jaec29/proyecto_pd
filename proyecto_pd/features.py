from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class SkewedYeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """Aplica Yeo-Johnson solo a las variables cuyo |skew| supera el umbral.

    Las variables sesgadas se determinan en `fit`, por lo que dentro de una
    validación cruzada la selección se hace con los datos de cada fold.
    """

    def __init__(self, variables, threshold=0.75):
        self.variables = variables
        self.threshold = threshold

    def fit(self, X, y=None):
        skewness = X[self.variables].skew()
        self.skewed_variables_ = skewness[skewness.abs() > self.threshold].index.tolist()
        self.transformer_ = (
            YeoJohnsonTransformer(variables=self.skewed_variables_).fit(X)
            if self.skewed_variables_
            else None
        )
        return self

    def transform(self, X):
        return self.transformer_.transform(X) if self.transformer_ else X.copy()
