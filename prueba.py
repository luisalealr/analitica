import os
import pickle
import pandas as pd
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

# Descargar recursos necesarios para nltk
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Definir archivos para datos y modelo
DATOS_USUARIO_FILE = 'datos_usuario.csv'
MODELO_FILE = 'modelo_entrenado.pkl'

# Verificar si el archivo de datos del usuario ya existe y cargarlo si es así
if os.path.exists(DATOS_USUARIO_FILE):
    df_datos_usuario = pd.read_csv(DATOS_USUARIO_FILE)
else:
    df_datos_usuario = pd.DataFrame(columns=['Categoria', 'Comentario', 'Etiqueta'])

# Verificar si el archivo del modelo entrenado ya existe y cargarlo si es así
if os.path.exists(MODELO_FILE):
    with open(MODELO_FILE, 'rb') as file:
        modelos = pickle.load(file)
else:
    modelos = {}

def preprocesar_texto(texto):
    doc = nlp(texto)
    palabras_procesadas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(palabras_procesadas)

def guardar_datos_usuario():
    # Guardar los datos ingresados por el usuario en un archivo CSV
    df_datos_usuario.to_csv(DATOS_USUARIO_FILE, index=False)

def entrenar_modelo():
    fases = df_datos_usuario['Categoria'].unique()
    modelos = {}
    
    for fase in fases:
        df_fase = df_datos_usuario[df_datos_usuario['Categoria'] == fase]
        
        if len(df_fase) < 2:
            print(f"No hay suficientes datos para entrenar el modelo para la fase {fase}.")
            continue

        X = df_fase['Comentario']
        y = df_fase['Etiqueta']

        stop_words_spanish = stopwords.words('spanish')
        vectorizer = TfidfVectorizer(stop_words=stop_words_spanish)
        X = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if len(y_train) == 0 or len(y_test) == 0:
            print(f"No hay suficientes datos para entrenar el modelo para la fase {fase}.")
            continue

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(f'Accuracy for {fase}:', accuracy_score(y_test, y_pred))
        
        modelos[fase] = (vectorizer, classifier)
    
    # Guardar el modelo entrenado
    with open(MODELO_FILE, 'wb') as file:
        pickle.dump(modelos, file)

    return modelos

def agregar_comentario(categoria, comentario, etiqueta):
    global df_datos_usuario
    nuevo_comentario = pd.DataFrame({'Categoria': [categoria], 'Comentario': [preprocesar_texto(comentario)], 'Etiqueta': [etiqueta]})
    df_datos_usuario = pd.concat([df_datos_usuario, nuevo_comentario], ignore_index=True)

def generar_respuesta_fase(modelo, texto_entrada):
    vectorizer, classifier = modelo
    texto_entrada_procesado = preprocesar_texto(texto_entrada)
    texto_entrada_vectorizado = vectorizer.transform([texto_entrada_procesado])
    prediccion = classifier.predict(texto_entrada_vectorizado)
    return prediccion[0]

def generar_respuesta_final():
    respuesta_final = []
    
    for fase in df_datos_usuario['Categoria'].unique():
        if fase in modelos:
            comentario = input(f"Ingrese un comentario para la fase {fase}: ")
            if comentario.strip():
                respuesta = generar_respuesta_fase(modelos[fase], comentario)
                respuesta_final.append(respuesta)
            else:
                respuesta_final.append("NO APLICA")
        else:
            respuesta_final.append("Modelo no entrenado para esta fase")
    
    respuesta_formateada = (
        f"El problema de la Universidad Francisco de Paula Santander, respecto al aumento de la desercion de estudiantes y una disminucion en la satisfaccion estudiantil a lo largo de los semestres estudiantiles, se presenta como causa de <<<<{respuesta_final[0]}(IDENTIFICACION DEL PROBLEMA)>>>>. "
        f"Se observan falencias en el siguiente aspecto <<<<{respuesta_final[1]}(PERSONAL)>>>>. "
        f"Ademas, existen <<<<{respuesta_final[2]}(METODO)>>>>. "
        f"Se identifica <<<<{respuesta_final[3]}(MAQUINA)>>>>. "
        f"Ademas, <<<<{respuesta_final[4]}(MATERIAL)>>>>. "
        f"Se ha notado <<<<{respuesta_final[5]}(MEDICION))>>>>. "
        f"Finalmente, <<<<{respuesta_final[6]}(ENTORNO)>>>>."
    )
    
    return respuesta_formateada

# Datos iniciales predefinidos
datos_iniciales = {
    'IDENTIFICACION DEL PROBLEMA': [
        "Alta tasa de deserción estudiantil.",
        "Problemas financieros.",
        "Problemas académicos.",
        "Problemas emocionales y de salud mental.",
        "Problemas de integración social."
    ],
    'ANALISIS DE PARTICIPACION': [
        "Estudiantes",
        "Profesores"
    ],
    'ANALISIS DE OBJETIVOS': [
        "Reducir la tasa de deserción estudiantil.",
        "Mejorar la asistencia financiera.",
        "Mejorar el apoyo académico y emocional.",
        "Mejorar la integración social de los estudiantes."
    ],
    'ANALISIS DE ALTERNATIVAS': [
        "Mejorar los programas de becas y ayudas financieras.",
        "Implementar programas de tutoría académica.",
        "Ofrecer servicios de consejería y apoyo emocional.",
        "Organizar actividades de integración social y construcción de comunidad."
    ],
    'DECLARACION DEL PROBLEMA': [
        "Descontento estudiantil por la calidad de los cursos.",
        "Disminución de la tasa de retención de estudiantes.",
        "Baja satisfacción estudiantil.",
        "Insatisfacción de los profesores con las condiciones laborales."
    ]
}

etiquetas_iniciales = [
    "problema central",
    "causas principales",
    "efectos",
    "participantes",
    "categorías",
   "caracterización",
    "objetivos principales",
    "relaciones 'medios - fines'",
    "alternativas propuestas",
    "criterios para elegir la mejor alternativa"
]

# Guardar los datos iniciales predefinidos en el DataFrame de datos del usuario
for categoria, datos in datos_iniciales.items():
    for dato in datos:
        agregar_comentario(categoria, dato, etiquetas_iniciales[datos_iniciales.index(datos)])

# Entrenar el modelo
entrenar_modelo()

# Generar y mostrar la respuesta final
respuesta_final = generar_respuesta_final()
print(respuesta_final)
