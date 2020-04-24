#CARREGANDO LIBRARIES
from sklearn import datasets
iris = datasets.load_iris()

iris.data

iris.target

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(iris.data,iris.target)

clf.feature_importances_

iris.feature_names

#underfited: generaliza tudo
#good fit: equilibrio entre generalização e decorar = o ideal
#overfitted: decora dados
iris.data[0]
clf.predict([iris.data[0]])

#metricas de avaliação
from sklearn import metrics
pred = clf.predict(iris.data)
metrics.accuracy_score(iris.target, pred)

# 1.0 = 100% (nada é 100%: esta errado!) é preciso uma parte do dataset para aprender e a outra parte para testar! imagine voce fazer uma prova com os mesmos exericios estudados: voce vai apenas testar sua memoria, nao a inteligencia
clf.fit(iris.data[:120], iris.target[:120])
pred = clf.predict(iris.data[120:])
metrics.accuracy_score(iris.target[120:], pred)

# o problema aqui? pegamos apenas os primeiros modelos, a maquina nao entendera ou aprendera como treinar os ultimos, afinal o dataset esta ordenado! 80% esta ok, mas se aparecer um target 2(versicolour) ele nao acertara pois nao aprendeu o que é versicolour!
#a maquina aprendeu o que é virginia e setosa, nao versicolour = BALANCEIE OS DADOS!
iris.target[120:],pred

#validação cruzada é avaliar os dados com ele mesmo = média da validation cross
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf,iris.data, iris.target, cv=10)
#cv é o numero de vezes da validação cruzada! ou seja, ensinou para 90% e treinou em 10%(cv)ele rodou 10x a forma randomica! por default rodaria 3! quanto maior o cv

import numpy as np
np.mean(score)

print(metrics.classification_report(iris.target[120:],pred))