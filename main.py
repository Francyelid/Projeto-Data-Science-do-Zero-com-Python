import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

arquivo = open('housing.csv')
linhas = csv.reader(arquivo)
for linha in linhas:
    print(linha)

# ANÁLISE EXPLORATÓRIA DOS DADOS
housing = pd.read_csv('housing.csv')
housing.head()
# Mostra os 5 primeiros reistros

# 10 features, sendo cada linha um distrito
# 20.639 amostras
housing.info()

# Única feature não-numérica
housing["ocean_proximity"].value_counts()

# Resumo numérico, "total_bedrooms" nos mostra alguns desafios
# SKILL NECESSÁRIA: Probabilidade/Estatística
housing.describe()

# SKILL NECESSÁRIA: Probabilidade/Estatística, Dataviz, Análise de Dados
housing.hist(bins=50, figsize=(20,15))
plt.show()

# PREPARANDO OS DADOS
# TRAIN/TEST SPLIT #usa-se o pip install scikit-learn para instalar a biblioteca

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

test_set.head()

housing["median_income"].hist()

# FEATURE ENGINEERING
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit #estratificação
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
 strat_train_set = housing.loc[train_index]
 strat_test_set = housing.loc[test_index]

# Dividimos os valores pelo tamanho do dataset e assim verificamos as proporções de "income_cat"
housing["income_cat"].value_counts() / len(housing)

# VISUALIZANDO OS DADOS 
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) #alpha se refere a densidade dos pontos

# A localização afeta o preço destes imóveis? 
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population",
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()

# Suficiente para uma correlação? valores entre -1 a 1, onde 1 significa CORRELAÇÃO POSITIVA FORTE
# Coeficiente de correlação de Pearson
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Correlação com PANDAS
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# LIMPEZA DE DADOS
# Removendo nossa target dos dados de TREINO
housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy() # realizamos uma cópia e armazenamos na variável "housing_labels"

# Verificando dados ausentes
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

# Utilizando Simple Imputer https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy="median")

# Limpando dados ausentes(missing values)
sample_incomplete_rows.drop("total_bedrooms", axis=1)  
median = housing["total_bedrooms"].median()

# Removemos a categoria não-numérica
housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)

# Transformaremos agora o conjunto TREINO com missing values substituidos pela mediana
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing.index)
housing_tr.head()

# PRÉ PROCESSAMENTO DA FEATURE CATEGORICA "ocean_proximity"
housing_cat = housing[['ocean_proximity']]
housing_cat.head()

# Lidando com dados NÃO-NUMÉRICOS 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded

# BINARIZAR
# É possível transformar inteiros em categorias, assim como categorias em números inteiros
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot

# ONE HOT ENCODER
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot

encoder.categories_

housing.columns

# TRANSFORMAÇÃO CUSTOMIZADA
# Maiores infos em https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html?highlight=baseestimator#sklearn.base.BaseEstimator
# Maiores infos em https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html?highlight=transformermixin#sklearn.base.TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin

# Índices das colunas com list comprehension
rooms_ix, bedrooms_ix, population_ix, household_ix = [list(housing.columns).index(col) for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(add_extra_features)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
 
 def __init__(self, attribute_names):
   self.attribute_names = attribute_names
 
 def fit(self, X, y=None):
   return self
 
 def transform(self, X):
   return X[self.attribute_names].values

housing_num_tr

housing_prepared

# MODELO PREDITIVO (TREINO)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse 

# Tentaremos com novo algoritmo, diferente da REGRESSÃO LINEAR

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# VALIDAÇÃO CRUZADA
# UTILIZANDO DECISIONTREEREGRESSOR
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# UTILIZANDO LINEARREGRESSION
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# TENTAREMOS COM OUTRO ALGORITMO: RANDOMFORESTREGRESSOR
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# AJUSTES DE PARÂMETROS
from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  print(np.sqrt(-mean_score), params)

# AVALIAÇÃO DOS MODELOS PREDITIVOS
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse

# REFERÊNCIAS: https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb