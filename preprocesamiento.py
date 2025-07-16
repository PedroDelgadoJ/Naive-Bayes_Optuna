import pandas as pd
import numpy as np
import re
import string

def convertir_minusculas(texto):
    """
    Convierte todo el texto a minúsculas.
    
    Args:
        texto (str): Texto a procesar
    
    Returns:
        str: Texto en minúsculas
    """
    return texto.lower()

def eliminar_puntuacion(texto):
    """
    Elimina signos de puntuación del texto.
    
    Args:
        texto (str): Texto a procesar
    
    Returns:
        str: Texto sin signos de puntuación
    """
    # Utilizamos una traducción para eliminar toda la puntuación
    translator = str.maketrans('', '', string.punctuation)
    return texto.translate(translator)

def eliminar_numeros(texto):
    """
    Elimina números del texto.
    
    Args:
        texto (str): Texto a procesar
    
    Returns:
        str: Texto sin números
    """
    return re.sub(r'\d+', '', texto)

def eliminar_espacios_extra(texto):
    """
    Elimina espacios en blanco múltiples y al principio/final.
    
    Args:
        texto (str): Texto a procesar
    
    Returns:
        str: Texto con espacios normalizados
    """
    return re.sub(r'\s+', ' ', texto).strip()

def preprocesar_df(df, columna, columna_nueva):
    funciones = [convertir_minusculas, eliminar_puntuacion, eliminar_numeros, eliminar_espacios_extra]
    for funcion in funciones:
        df[columna_nueva] = df[columna].apply(funcion)
    return df