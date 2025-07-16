from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import  TfidfVectorizer



# Función para crear vectorizador y transformar datos
def crear_tfidf(df_train, df_test, columna_texto, logs = True, **params):
    '''
    Params:
        df_train: DataFrame de entrenamiento
        df_test: DataFrame de prueba
        columna_texto: Nombre de la columna que contiene el texto procesado
        min_df: Frecuencia mínima de documento para incluir un término
        max_df: Frecuencia máxima de documento para incluir un término
        norm: Normalización aplicada ('l1', 'l2' o None)
        use_idf:
            - Si es True, se aplica la ponderación IDF (Inverse Document Frequency).
            - Esto penaliza palabras comunes (como "el", "y", "de") que aparecen en muchos documentos.
            - Si es False, solo se usa TF (frecuencia del término en el documento), como un CountVectorizer.

        smooth_idf:
            - Si es True, se le suma 1 al numerador y al denominador en el cálculo del IDF.
            - Esto evita problemas cuando un término aparece en todos los documentos (división por cero).
            - Su fórmula sería: idf(t) = log((1 + n) / (1 + df(t))) + 1

        sublinear_tf:
            - Si es True, se aplica una escala logarítmica a la frecuencia del término: tf = 1 + log(tf)
            - Esto reduce el impacto de palabras muy frecuentes dentro del mismo documento.
    '''
    vectorizador = TfidfVectorizer(**params)
    X_train = vectorizador.fit_transform(df_train[columna_texto])
    X_test = vectorizador.transform(df_test[columna_texto])

    if logs == True:
        # Obtenemos información sobre el vocabulario
        vocabulario = vectorizador.get_feature_names_out()
        
        print(f"- Tamaño del vocabulario: {len(vocabulario)} términos")
        print(f"- Dimensiones matriz train: {X_train.shape}")
        print(f"- Dimensiones matriz test: {X_test.shape}")
        print(f"- Densidad matriz train: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.5f}")
        

    return vectorizador, X_train, X_test, vocabulario.shape

# Función para entrenar y evaluar
def probar_modelo(df_train, df_test, y_train, y_test, columna_texto, tfidf_params, alpha):
    vectorizador, X_train_tfidf, X_test_tfidf = crear_tfidf(
        df_train, df_test, columna_texto, **tfidf_params
    )
    # alpha: Suavizado de Laplace, para tener en cuenta palabras no vistas en el entrenamiento.
    modelo = MultinomialNB(alpha=alpha)
    modelo.fit(X_train_tfidf, y_train)
    
    acc_train = modelo.score(X_train_tfidf, y_train)
    acc_test = modelo.score(X_test_tfidf, y_test)
    
    return acc_train, acc_test