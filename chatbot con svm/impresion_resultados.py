
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# Cargar el vectorizador
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Cargar el modelo SVM
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Ver las características del vectorizador
print("Características del Vectorizador (vocabulario):")
print(vectorizer.get_feature_names_out())

# Ver los parámetros del modelo SVM
print("\nParámetros del Modelo SVM:")
print(svm_model.get_params())

# Ver los coeficientes del modelo SVM
print("\nCoeficientes del Modelo SVM:")
print(svm_model.coef_)

import pandas as pd

# Obtener el vocabulario del vectorizador
vocabulario = vectorizer.get_feature_names_out()

# Obtener los coeficientes del modelo para cada clase
coef = svm_model.coef_.toarray()

# Sumar los valores absolutos de los coeficientes para cada palabra en todas las clases
total_abs_coef = np.abs(coef).sum(axis=0)

# Crear un DataFrame para visualizar las características y sus coeficientes
df = pd.DataFrame({'Feature': vocabulario, 'TotalAbsCoefficient': total_abs_coef})

# Ordenar por valor absoluto del coeficiente total
df = df.sort_values('TotalAbsCoefficient', ascending=False)

# Mostrar las palabras más importantes en general
print("Palabras más importantes en general:")
print(df.head(20))

# Crear una nube de palabras combinando todas las clases
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Feature']))

# Mostrar la nube de palabras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



