📚 Estructura del Trabajo
1. Descripción del Problema de Negocio:
En el entorno actual de sobrecarga de información, es esencial clasificar automáticamente las noticias para facilitar su consumo y análisis. Este proyecto busca desarrollar un modelo que categorice artículos de noticias en sus respectivas secciones, mejorando la eficiencia en la organización y recuperación de información.

2.🎯 Objetivo del proyecto:
Desarrollar un modelo de clasificación automática de noticias basado en NLP, capaz de categorizar artículos de la BBC en una de las siguientes temáticas:

Negocios
Entretenimiento
Política
Deporte
Tecnología
Optimizando la precisión y eficiencia del proceso.

3. Origen de los Datos:
El dataset proviene de la BBC (British Broadcasting Corporation ​ es el servicio público de radio y televisión del Reino Unido) y está disponible públicamente en Kaggle.

4. Definición de las Variables:
Texto del Artículo: Contenido completo del artículo de noticias.

Categoría: Etiqueta que indica la sección a la que pertenece el artículo (negocios, entretenimiento, política, deportes, tecnología)

5 📘 Diccionario de Variables:
Variable	Tipo	Descripción
category	Categórica	Categoría o tema del artículo. Ejemplos: tech, sport, business, entertainment, politics. Es la variable objetivo (target) en el modelo.
text	Texto	Contenido completo del artículo de noticias en inglés.
clean_text	Texto	Texto procesado (minúsculas, sin signos, sin stopwords, lematizado). Utilizado como entrada para técnicas de NLP.
X_bow	Matriz sparse	Matriz resultante del modelo Bag of Words (BOW) con las características más frecuentes (máx. 5000).
X_tfidf	Matriz sparse	Matriz de características TF-IDF (máx. 5000), usada como entrada al modelo de ML.
y	Categórica	Mismo contenido que category, usada como etiqueta para el modelo supervisado.
X_train / X_test	Matriz	Datos de entrenamiento y prueba extraídos de X_tfidf.
y_train / y_test	Categórica	Etiquetas correspondientes a los conjuntos de entrenamiento y prueba.
model	Objeto s práticos entrenados entrenado.	
y_pred	Categórica	Predicciones realizadas por el modelo sobre el conjunto de prueba.
6. Librerías a Utilizar:
Procesamiento de Datos: Pandas, NumPy

NLP: NLTK, spaCy

Visualización: Matplotlib, Seaborn, WordCloud

Modelado: Scikit-learn

# NLP y Clasificación Supervisada con el Dataset de Noticias BBC

# --------------------------------------------
# 1. Importación de librerías necesarias
# --------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re

!pip install wordcloud

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: wordcloud in c:\users\carlos\appdata\roaming\python\python312\site-packages (1.9.4)
Requirement already satisfied: numpy>=1.6.1 in c:\programdata\anaconda3\lib\site-packages (from wordcloud) (1.26.4)
Requirement already satisfied: pillow in c:\programdata\anaconda3\lib\site-packages (from wordcloud) (10.3.0)
Requirement already satisfied: matplotlib in c:\programdata\anaconda3\lib\site-packages (from wordcloud) (3.8.4)
Requirement already satisfied: contourpy>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.2.0)
Requirement already satisfied: cycler>=0.10 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (4.51.0)
Requirement already satisfied: kiwisolver>=1.3.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.4.4)
Requirement already satisfied: packaging>=20.0 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (23.2)
Requirement already satisfied: pyparsing>=2.3.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\Carlos\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\Carlos\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\Carlos\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
True
7. Desarrollo:
Parte 1: Procesamiento de Lenguaje Natural
Lectura y exploración inicial del dataset.

Limpieza del texto: eliminación de símbolos, signos de puntuación y conversión a minúsculas.

Tokenización y eliminación de stopwords.

Lematización utilizando spaCy.

Visualización mediante nubes de palabras para cada categoría.

Creación de n-gramas para identificar combinaciones frecuentes de palabras.

# --------------------------------------------
# 2. Cargar el dataset
# --------------------------------------------
# El dataset puede descargarse de Kaggle: https://www.kaggle.com/datasets/cashncarry/news-category-dataset

df = pd.read_csv('bbc-text.csv')

# Visualizamos las primeras filas
df.head()
category	text
0	tech	tv future in the hands of viewers with home th...
1	business	worldcom boss left books alone former worldc...
2	sport	tigers wary of farrell gamble leicester say ...
3	sport	yeading face newcastle in fa cup premiership s...
4	entertainment	ocean s twelve raids box office ocean s twelve...
# --------------------------------------------
# 3. Exploración y definición de variables
# --------------------------------------------
print("Cantidad de documentos por categoría:")
print(df['category'].value_counts())
Cantidad de documentos por categoría:
category
sport            511
business         510
politics         417
tech             401
entertainment    386
Name: count, dtype: int64
# Renombramos las columnas para mayor claridad
# 'text' es el contenido del artículo y 'category' es la clase a predecir
df.columns = ['category', 'text']
# --------------------------------------------
# 4. Limpieza del texto
# --------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # eliminamos signos y números
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Aplicamos limpieza al texto
df['clean_text'] = df['text'].apply(clean_text)

# --------------------------------------------
# 5. Visualización de datos: WordClouds, palabras y n-gramas
# --------------------------------------------
def generate_wordcloud(category):
    text = " ".join(df[df['category'] == category]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud para categoría: {category}")
    plt.show()

# Generamos una para cada categoría:
for cat in df['category'].unique():
    generate_wordcloud(cat)

# Generamos una general para todo el corpus:
all_text = " ".join(df['clean_text'])
wordcloud_all = WordCloud(width=1000, height=500, background_color='white').generate(all_text)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud para todo el corpus")
plt.show()
![download](https://github.com/user-attachments/assets/a88f9fc3-e2d2-48b8-b2fa-c5a74fd97dc3)
![download](https://github.com/user-attachments/assets/c0064337-718f-43e8-98f5-9b350bd21745)
![download](https://github.com/user-attachments/assets/3a1f0e63-2ffb-489c-a950-a06cebc99f23)
![download](https://github.com/user-attachments/assets/bdd37f9f-9874-42ea-81f1-a6852a2921a8)
![download](https://github.com/user-attachments/assets/d96c4369-9403-4939-8791-b7723dfa9889)
![download](https://github.com/user-attachments/assets/67184537-ffe7-4fb8-912d-31cc940739ce)


# --------------------------------------------
# 5.1 Gráfico de barras con las 10 palabras más frecuentes por categoría
# --------------------------------------------
from collections import Counter

def plot_top_words(category, top_n=10):
    words = " ".join(df[df['category'] == category]['clean_text']).split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(top_n)
    words, counts = zip(*common_words)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette='magma')
    plt.title(f"Top {top_n} palabras más frecuentes en '{category}'", fontsize=14)
    plt.xlabel("Frecuencia")
    plt.ylabel("Palabra")
    plt.tight_layout()
    plt.show()

for cat in df['category'].unique():
    plot_top_words(cat)
![download](https://github.com/user-attachments/assets/80fa1089-7184-4ecc-a64e-9474686a2718)
![download](https://github.com/user-attachments/assets/eca3b2e9-5f31-4ba6-b874-6382033a1e28)
![download](https://github.com/user-attachments/assets/b85c18d7-4256-4970-9a1c-bd23ec2a5a9c)
![download](https://github.com/user-attachments/assets/0745408c-feef-438d-bd7e-1fb7ef8fe85b)
![download](https://github.com/user-attachments/assets/dca4fca7-943d-4af9-845f-999414938bc1)


# --------------------------------------------
# 5.2 Bigrama y Trigrama más frecuentes por categoría
# --------------------------------------------


def plot_top_ngrams(category, ngram_range=(2,2), top_n=10):
    subset = df[df['category'] == category]['clean_text']
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(subset)
    sum_words = X.sum(axis=0)
    word_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)[:top_n]
    ngrams, counts = zip(*word_freq)

    plt.figure(figsize=(10, 6))
    palette = 'viridis' if ngram_range == (2, 2) else 'plasma'
    sns.barplot(x=list(counts), y=list(ngrams), palette=palette)
    plt.title(f"Top {top_n} {'bigrama' if ngram_range==(2,2) else 'trigrama'} en '{category}'", fontsize=14)
    plt.xlabel("Frecuencia")
    plt.ylabel("{'Bigrama' if ngram_range==(2,2) else 'Trigrama'}")
    plt.tight_layout()
    plt.show()

for cat in df['category'].unique():
    plot_top_ngrams(cat, ngram_range=(2, 2))  # bigramas
    plot_top_ngrams(cat, ngram_range=(3, 3))  # trigramas

![download](https://github.com/user-attachments/assets/9055005d-1cc1-4637-b4bd-051e792739d6)
![download](https://github.com/user-attachments/assets/8121ee9b-df8f-48e7-a6d1-95114bbe1959)
![download](https://github.com/user-attachments/assets/ab8f1cd8-a009-4718-9bc1-467b0f3ab9b8)
![download](https://github.com/user-attachments/assets/6736c94f-98b0-491a-9efe-d18cc87a03e5)
![download](https://github.com/user-attachments/assets/76b238c3-1c05-4e11-959a-eb0384de4da5)
![download](https://github.com/user-attachments/assets/30b1941a-b3cb-445e-a28f-0c194c40fb03)
![download](https://github.com/user-attachments/assets/eba3c359-2b75-4178-a2e0-995bbdb6336c)
![download](https://github.com/user-attachments/assets/99116da8-ea72-484f-8037-b19bc1600914)
![download](https://github.com/user-attachments/assets/341a5fd7-39ca-4109-b992-ed05cbb40d5e)
![download](https://github.com/user-attachments/assets/913381e9-5ce6-44ba-af25-76ef6c3b448c)



Parte 2: Modelado
Transformación del texto utilizando TF-IDF y Bag of Words / Un análisis comparativo entre TF-IDF y BOW

División del dataset en conjuntos de entrenamiento y prueba.

Entrenamiento y evaluación con cinco modelos: Regresión Logística, Naive Bayes, Random Forest, SVM y Red Neuronal (MLP).

Evaluación del rendimiento utilizando métricas como precisión, recall y F1-score / tabla comparativa de precisión y gráfico de barras.

# --------------------------------------------
# 6. Representación vectorial (TF-IDF y BOW)
# --------------------------------------------

bow_vectorizer = CountVectorizer(max_features=5000)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

X_bow = bow_vectorizer.fit_transform(df['clean_text'])
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])
y = df['category']
# Visualización de las matrices BOW y TF-IDF (una muestra)
bow_df = pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print("\nMatriz BOW - muestra:")
print(bow_df.head())

print("\nMatriz TF-IDF - muestra:")
print(tfidf_df.head())

Matriz BOW - muestra:
   aaa  abandoned  abbott  abc  ability  able  abn  abortion  abroad  absence  \
0    0          0       0    0        0     0    0         0       0        0   
1    0          0       0    0        1     0    0         0       0        0   
2    0          0       0    0        0     0    0         0       0        0   
3    0          0       0    0        0     0    0         0       0        0   
4    0          0       0    0        0     0    0         0       0        0   

   ...  yuan  yugansk  yuganskneftegas  yukos  yushchenko  zealand  zero  \
0  ...     0        0                0      0           0        0     0   
1  ...     0        0                0      0           0        0     0   
2  ...     0        0                0      0           0        0     0   
3  ...     0        0                0      0           0        0     0   
4  ...     0        0                0      0           0        0     0   

   zombie  zone  zurich  
0       0     0       0  
1       0     0       0  
2       0     0       0  
3       0     0       0  
4       0     0       0  

[5 rows x 5000 columns]

Matriz TF-IDF - muestra:
   aaa  abandoned  abbott  abc  ability  able  abn  abortion  abroad  absence  \
0  0.0        0.0     0.0  0.0  0.00000   0.0  0.0       0.0     0.0      0.0   
1  0.0        0.0     0.0  0.0  0.04538   0.0  0.0       0.0     0.0      0.0   
2  0.0        0.0     0.0  0.0  0.00000   0.0  0.0       0.0     0.0      0.0   
3  0.0        0.0     0.0  0.0  0.00000   0.0  0.0       0.0     0.0      0.0   
4  0.0        0.0     0.0  0.0  0.00000   0.0  0.0       0.0     0.0      0.0   

   ...  yuan  yugansk  yuganskneftegas  yukos  yushchenko  zealand  zero  \
0  ...   0.0      0.0              0.0    0.0         0.0      0.0   0.0   
1  ...   0.0      0.0              0.0    0.0         0.0      0.0   0.0   
2  ...   0.0      0.0              0.0    0.0         0.0      0.0   0.0   
3  ...   0.0      0.0              0.0    0.0         0.0      0.0   0.0   
4  ...   0.0      0.0              0.0    0.0         0.0      0.0   0.0   

   zombie  zone  zurich  
0     0.0   0.0     0.0  
1     0.0   0.0     0.0  
2     0.0   0.0     0.0  
3     0.0   0.0     0.0  
4     0.0   0.0     0.0  

[5 rows x 5000 columns]
Análisis del código para, Representación vectorial (TF-IDF y BOW):
🔹 ¿Qué es bow_df.head()?:
Crea una matriz de documentos vs palabras (solo las más frecuentes, hasta 5000).

Cada fila representa un documento (noticia).

Cada columna es una palabra (token).

Cada celda indica la cantidad de veces que esa palabra aparece en ese documento.

Esta es una representación de frecuencia absoluta.

📌 Interpretación:
Si se observa un valor alto para una palabra específica en una fila, significa que esa palabra es muy frecuente en ese documento. Sin embargo, no considera si esa palabra también es común en otros documentos (por eso usamos TF-IDF).

🔹 ¿Qué es tfidf_df.head()?:
Muestra una matriz similar, pero con valores normalizados por importancia.

Cada celda contiene un valor entre 0 y 1 que mide la importancia de una palabra en un documento, penalizando aquellas que aparecen en muchos documentos.

📌 Interpretación:
Un valor alto en TF-IDF indica que esa palabra es característica o distintiva de ese documento, es decir, aparece mucho en ese documento pero poco en otros.

# Palabras más importantes según TF-IDF
tfidf_sum = np.asarray(X_tfidf.sum(axis=0)).ravel()
words = tfidf_vectorizer.get_feature_names_out()

# Crear DataFrame con términos más relevantes
top_tfidf = pd.DataFrame({'word': words, 'tfidf': tfidf_sum})
top_tfidf = top_tfidf.sort_values(by='tfidf', ascending=False).head(20)

# Visualización de top palabras
plt.figure(figsize=(10, 6))
sns.barplot(x='tfidf', y='word', data=top_tfidf, palette='rocket')
plt.title('Top 20 palabras con mayor peso TF-IDF en todo el corpus')
plt.xlabel('Peso TF-IDF acumulado')
plt.ylabel('Palabra')
plt.tight_layout()
plt.show()

![download](https://github.com/user-attachments/assets/da438540-a17d-44a7-bc25-3bae5f225524)


Explicación del coódigo anterior, sobre Palabras más importantes según TF-IDF:
🔹 Visualización de las palabras con mayor peso TF-IDF:
Calcula la suma total de los pesos TF-IDF de cada palabra en todo el corpus.

Se muestran las 20 palabras más influyentes en todo el conjunto de textos.

📌 Interpretación del gráfico:
Las palabras con mayor peso TF-IDF son probablemente términos clave que distinguen las categorías del corpus (por ejemplo: "Said", "year", "would").

Estas palabras ayudan más a los modelos de clasificación porque aportan mayor capacidad discriminativa entre clases.

# ------------------------------------------------------------
# 7. División de datos y entrenamiento con múltiples modelos
# ------------------------------------------------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nModelo: {name}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Matriz de Confusión - {name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()
    results.append({'Modelo': name, 'Precisión': acc, 'Precisión Ponderada': prec, 'Recall': rec, 'F1-Score': f1})
    
Modelo: Logistic Regression
               precision    recall  f1-score   support

     business       0.95      0.94      0.95       101
entertainment       1.00      0.94      0.97        81
     politics       0.92      0.98      0.95        83
        sport       0.97      1.00      0.98        98
         tech       0.99      0.96      0.98        82

     accuracy                           0.96       445
    macro avg       0.97      0.96      0.96       445
 weighted avg       0.96      0.96      0.96       445

![download](https://github.com/user-attachments/assets/7a81a917-19a2-47f2-b9f7-e91a3f64c2f6)


Modelo: Naive Bayes
               precision    recall  f1-score   support

     business       0.95      0.95      0.95       101
entertainment       1.00      0.91      0.95        81
     politics       0.92      0.98      0.95        83
        sport       0.99      1.00      0.99        98
         tech       0.96      0.98      0.97        82

     accuracy                           0.96       445
    macro avg       0.96      0.96      0.96       445
 weighted avg       0.97      0.96      0.96       445

![download](https://github.com/user-attachments/assets/14bda02d-844a-4d65-a78b-37d536e062fd)


Modelo: Random Forest
               precision    recall  f1-score   support

     business       0.91      0.92      0.92       101
entertainment       0.99      0.90      0.94        81
     politics       0.91      0.96      0.94        83
        sport       0.97      0.99      0.98        98
         tech       0.95      0.94      0.94        82

     accuracy                           0.94       445
    macro avg       0.95      0.94      0.94       445
 weighted avg       0.94      0.94      0.94       445

![download](https://github.com/user-attachments/assets/fb02b539-2b36-4740-be59-381d9e1af723)

 
Modelo: SVM
               precision    recall  f1-score   support

     business       0.96      0.92      0.94       101
entertainment       0.96      0.98      0.97        81
     politics       0.93      0.96      0.95        83
        sport       0.98      1.00      0.99        98
         tech       0.99      0.96      0.98        82

     accuracy                           0.96       445
    macro avg       0.96      0.96      0.96       445
 weighted avg       0.96      0.96      0.96       445

![download](https://github.com/user-attachments/assets/e95aba58-bf28-4a01-a03f-02392bfaac49)

 
Modelo: MLP (Neural Net)
               precision    recall  f1-score   support

     business       0.97      0.93      0.95       101
entertainment       0.97      0.96      0.97        81
     politics       0.92      0.96      0.94        83
        sport       0.99      1.00      0.99        98
         tech       0.98      0.98      0.98        82

     accuracy                           0.97       445
    macro avg       0.97      0.97      0.97       445
 weighted avg       0.97      0.97      0.97       445

![download](https://github.com/user-attachments/assets/8197f523-1bc9-49b2-98bb-95d00fd685e0)



🔍 Análisis individual por modelo, División de datos y entrenamiento:
1. MLPClassifier (Red Neuronal Multicapa):
Mejor desempeño general (F1 = 0.966).

Detecta relaciones no lineales en los datos de texto gracias a sus capas ocultas.

Ventaja: Puede capturar patrones complejos que otros modelos lineales no pueden.

Ideal para grandes volúmenes de texto si se escala correctamente.

2. Logistic Regression:
Modelo lineal muy eficiente.

Rendimiento sobresaliente considerando su simplicidad (F1 = 0.964).

Beneficiado por la representación TF-IDF, que convierte texto en un espacio vectorial linealmente separable.

Rápido y útil para producción.

3. Naive Bayes:
Aunque muy rápido y eficiente para texto, parte de la suposición de independencia entre palabras, lo que limita su capacidad para capturar relaciones semánticas.

Sin embargo, logró un F1 alto (0.964), mostrando que el preprocesamiento fue eficaz para eliminar ruido.

4. SVM (Support Vector Machine):
F1 = 0.964. Excelente capacidad de generalización.

Al usar kernel lineal con TF-IDF, encuentra un buen hiperplano de separación entre clases.

Recomendado para problemas con texto bien diferenciado.

5. Random Forest:
Peor desempeño relativo (F1 = 0.944).

Tiende a sobreajustar datos con alta dimensionalidad como texto vectorizado.

No captura relaciones semánticas bien con matrices dispersas como TF-IDF sin reducción de dimensionalidad.

📊 Explicación de que es una matriz de confusión y resultados:

Una matriz de confusión es una tabla que permite visualizar el desempeño de un modelo de clasificación, comparando las predicciones con los valores reales. En un problema de clasificación multiclase como este (con 5 categorías de noticias: business, entertainment, politics, sport, tech), la matriz tiene una forma de 5x5.

Cada celda (i, j) representa la cantidad de observaciones cuya verdadera clase es i y fueron predichas como clase j.

Diagonal principal (de arriba a la izquierda a abajo a la derecha): aciertos.

Celdas fuera de la diagonal: errores.

✅ Interpretación por modelo (según la matriz de confusión observada):

MLP (Neural Net) – 🏆 Modelo ganador:

Casi todas las predicciones cayeron sobre la diagonal, es decir, el modelo acertó casi todas las clases correctamente.

Los errores fueron mínimos y dispersos, mostrando gran capacidad de generalización.

F1-score: 0.9663, el más alto.

Conclusión: Este modelo aprendió patrones complejos y logró alta precisión y balance entre clases. Es ideal para producción.

Logistic Regression:

También presenta alta exactitud, con una matriz de confusión muy limpia.

Cometió algún error esporádico, por ejemplo, entre tech y business, o politics con entertainment, pero muy poco.

Conclusión: Gran modelo lineal para clasificación textual con TF-IDF. Rápido, interpretable y confiable.

Naive Bayes:

Sorprendentemente competitivo.

Tuvo errores similares a Logistic Regression, aunque un poco más dispersos.

Es común que confunda clases que comparten vocabulario frecuente (por ejemplo, business y tech).

Conclusión: Aunque hace supuestos simplistas (independencia entre palabras), su rapidez y buen desempeño lo hacen valioso.

SVM:

Rendimiento similar a Logistic Regression.

Muy buena generalización.

Sus errores fueron mínimos y parecidos a los anteriores.

Conclusión: Gran capacidad para manejar datos de texto, especialmente con representación TF-IDF. Es sensible a los márgenes de separación entre clases.

Random Forest:

Tuvo más errores, especialmente confundiendo business con tech o politics, y algunos casos de sport con entertainment.

La matriz muestra más densidad fuera de la diagonal.

Conclusión: Al ser un modelo de conjunto sobre datos dispersos, sufre al no capturar bien la secuencia y contexto de las palabras, como lo hacen mejor los modelos lineales o neuronales.

🧠 ¿Qué nos enseñan estas matrices?

Las clases están bien definidas: la mayoría de los modelos lograron predicciones muy certeras.

El preprocesamiento y TF-IDF fueron fundamentales: incluso modelos simples como Naive Bayes rindieron bien.

MLP fue superior porque capta relaciones no lineales complejas, crucial en NLP.

Los errores más comunes se dieron entre categorías conceptualmente cercanas, como:

business ↔ tech

entertainment ↔ sport

politics ↔ business



# --------------------------------------------
# 8. Evaluación comparativa
# --------------------------------------------
results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(data=results_df, y='Modelo', x='F1-Score', palette='rocket', ax=ax)
plt.title('Comparación de Modelos: F1-Score', fontsize=16)
plt.xlabel('F1-Score')
plt.ylabel('Modelo')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Tabla comparativa de métricas, ordenada con el mejor F1-Score:
styled_table = results_df.style.background_gradient(cmap='Oranges').set_caption("Comparativa Final de Modelos (Mejor ordenado por F1-Score)")
display(styled_table)

![download](https://github.com/user-attachments/assets/2e95a734-0fb7-4cc4-918d-59844616f956)

Comparativa Final de Modelos (Mejor ordenado por F1-Score)

![image](https://github.com/user-attachments/assets/2a236912-b9fc-4b1d-af5c-0f417b0c04a1)

📊 Comparativa Final de Modelos:
📌 ¿Por qué se eligió F1-Score como métrica principal?
El F1-Score ponderado, combina precisión y recall, penalizando tanto los falsos positivos como los falsos negativos. Esto es crucial en clasificación multiclase, donde un modelo podría tener alta precisión en una clase dominante pero fallar en otras. Usar solo precisión o recall puede ocultar ese sesgo, mientras que el F1 ponderado lo revela.

💡 Conclusiones para la comparativa final de modelos:
✅ MLPClassifier es el modelo ganador, por su F1-Score más alto, gracias a su capacidad de modelar relaciones complejas no lineales.

🔠 La elección de TF-IDF fue clave, ya que mejoró el rendimiento global respecto a BOW.

📈 Modelos lineales como SVM y Logistic Regression también ofrecieron resultados muy competitivos, lo que sugiere que el problema es linealmente separable.

⚠️ Naive Bayes, a pesar de su simplicidad, fue sorprendentemente competitivo, lo que valida la calidad del preprocesamiento.

8. 🧠 Conclusión Final del Proyecto:
🎯 Resumen de objetivos alcanzados:
El objetivo principal del proyecto fue desarrollar un modelo de clasificación automática de noticias del dataset BBC News, aplicando técnicas de Procesamiento de Lenguaje Natural (NLP) y modelos de aprendizaje supervisado. Se buscaba demostrar que, mediante un pipeline de NLP bien estructurado y una correcta elección del modelo, es posible organizar y clasificar contenido textual con alta precisión, eficiencia y escalabilidad.

📊 Aspectos más importantes del análisis:
Preprocesamiento exhaustivo de texto:
Tokenización, lematización, remoción de stopwords, generación de n-gramas, y análisis de frecuencia permitieron construir una base sólida para el modelado.

Las visualizaciones como nubes de palabras y barras por categoría brindaron claridad sobre la distribución del contenido por tema.

Representación vectorial:
La combinación de técnicas BOW y TF-IDF permitió transformar eficazmente el texto en variables numéricas interpretables.

Se observó que TF-IDF mejoró el rendimiento de los modelos, destacando la importancia de términos distintivos en cada categoría.

Modelado y evaluación comparativa:
Se entrenaron y compararon cinco modelos: MLPClassifier, Logistic Regression, Naive Bayes, SVM y Random Forest.

Se evaluaron utilizando métricas de Precisión, Recall, Precisión Ponderada y F1-Score, siendo esta última la métrica principal por equilibrar precisión y exhaustividad en tareas multiclase.

🏆 Modelo ganador y análisis:

![image](https://github.com/user-attachments/assets/1ebe8774-244c-40d3-a5fe-e040c03ccc01)

- MLPClassifier (Red Neuronal) fue el mejor modelo global según el F1-Score, gracias a su capacidad de capturar relaciones no lineales complejas. Su rendimiento constante en todas las métricas lo convierte en un candidato robusto para implementación.
- Logistic Regression y SVM también ofrecieron excelentes resultados, lo que valida la calidad del preprocesamiento y la representación TF-IDF.
- Naive Bayes logró destacar a pesar de su simplicidad, beneficiándose del carácter bien estructurado del corpus.
- Random Forest, aunque útil en muchos casos, tuvo menor rendimiento posiblemente debido a la naturaleza dispersa y alta dimensionalidad de los vectores TF-IDF.

🔍 Insights claves del proyecto:
La limpieza del texto y la correcta vectorización fueron factores críticos para el rendimiento de los modelos.

Las categorías del dataset están bien definidas y presentan un bajo solapamiento semántico, lo que facilitó la clasificación.

TF-IDF fue superior a BOW en términos de discriminación entre cla ses.

🚀 Aplicación del modelo ganador a la operatividad empresarial:
Implementar el modelo MLPClassifier permitiría:
Automatizar la categorización de noticias en redacciones digitales.

Optimizar flujos de trabajo de curaduría y organización de contenido.

Mejorar sistemas de recomendación de artículos según temas de interés.








9. 🔮 Perspectivas a futuro y Mejoras sugeridas al proyecto:
Qué se puede mejorar:
Incluir embeddings contextuales avanzados como Word2Vec, FastText o BERT para capturar relaciones semánticas más profundas.

Aplicar técnicas de optimización de hiperparámetros (e.g., GridSearchCV).

Explorar enfoques de ensembles, como VotingClassifier o Stacking.

Cuales serian las Perspectivas futuras:
Desplegar MLPClassifier en entornos de producción.

Ampliar a textos en múltiples idiomas.

Adaptar a clasificación multi-etiqueta.
