import optuna
import tf_idf
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import accuracy_score


def objective(trial, df_train, df_test, y_train, y_test, columna_texto, resultados_trials, variable ='mean'):
    # Hiperparámetros de TF-IDF
    min_df = trial.suggest_int("min_df", 1, 4000, step= 50)
    max_df = trial.suggest_float("max_df", 0.1, 1.0)
    norm = trial.suggest_categorical("norm", ["l1", "l2", None])
    use_idf = trial.suggest_categorical("use_idf", [True, False])
    smooth_idf = trial.suggest_categorical("smooth_idf", [True, False])
    sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])

    # Hiperparámetro del clasificador
    alpha = trial.suggest_float("alpha", 1e-2, 5.0, log=True)

    try:
        # Vectorización
        tfidf_params = {
            'min_df': min_df,
            'max_df': max_df,
            'norm': norm,
            'use_idf': use_idf,
            'smooth_idf': smooth_idf,
            'sublinear_tf': sublinear_tf,
        }

        vectorizador, X_train_tfidf, X_test_tfidf, vocabulario = tf_idf.crear_tfidf(
            df_train, df_test, columna_texto= columna_texto, **tfidf_params
        )

        modelo = MultinomialNB(alpha=alpha)
        modelo.fit(X_train_tfidf, y_train)

        # Precisión
        y_train_pred = modelo.predict(X_train_tfidf)
        y_test_pred = modelo.predict(X_test_tfidf)

        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        mean_acc = (acc_train + acc_test) / 2

        # Guardar los resultados del trial
        resultados_trials.append({
            'trial_number': trial.number,
            'alpha': alpha,
            'min_df': min_df,
            'max_df': max_df,
            'norm': norm,
            'use_idf': use_idf,
            'smooth_idf': smooth_idf,
            'sublinear_tf': sublinear_tf,
            'size': vocabulario,
            'accuracy_train': acc_train,
            'accuracy_test': acc_test,
            'accuracy_mean': mean_acc,
        })
        if variable = 'train':
            return acc_train
        elif variable = 'test':
            return acc_test
        elif variable = 'mean':
            return mean_acc

    except Exception as e:
        print(f"Error en trial {trial.number}: {e}")
        return 0.0


