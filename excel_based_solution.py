import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.callbacks import Callback

# Fonction pour prétraiter les données
def preprocess(uploaded_file, date_column, colis_column):
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'xlsx':
        df = pd.read_excel(uploaded_file, index_col=None)
    elif file_type == 'csv':
        df = pd.read_csv(uploaded_file, index_col=None)
    else:
        return "Format de fichier non pris en charge"
    
    if date_column not in df.columns or colis_column not in df.columns:
        raise ValueError(f"Le fichier doit contenir les colonnes '{date_column}' et '{colis_column}'")
    
    df[date_column] = pd.to_datetime(df[date_column])
    df[colis_column] = df[colis_column].fillna(0).astype(int)
    
    return df

# Fonction pour créer des séquences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

class ProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.progress_bar = st.progress(0)
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)

def train_model(X_train, y_train, epochs):
    model = Sequential([
        Input((X_train.shape[1], 1)),
        LSTM(50, activation='relu', return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    progress_callback = ProgressCallback(epochs)
    
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=32, 
        validation_split=0.1, 
        verbose=0,
        callbacks=[progress_callback]
    )
    return model, history


def convert_dates(df, date_column):
    if df[date_column].dtype == 'object':
        df[date_column] = pd.to_datetime(df[date_column])
    return df

def plot_learning_curves(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], name="Perte d'entraînement"))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Perte de validation'))
    fig.update_layout(title="Courbes d'apprentissage",
                      xaxis_title='Époques',
                      yaxis_title='Perte',
                      legend_title='Légende')
    return fig

def plot_input_data(df, date_column, colis_column):
    df = convert_dates(df, date_column)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_column], y=df[colis_column], mode='lines', name='Exportations réelles'))
    fig.update_layout(title='Données d\'exportation',
                      xaxis_title='Date',
                      yaxis_title='Exportations',
                      legend_title='Légende')
    return fig

def plot_predictions(df, train_predict, test_predict, future_predict, train_size, seq_length, date_column, colis_column):
    df = convert_dates(df, date_column)
    fig = go.Figure()

    # Données réelles
    fig.add_trace(go.Scatter(x=df[date_column], y=df[colis_column], mode='lines', name='Réel', line=dict(color='blue')))

    # Prédictions d'entraînement
    train_dates = df[date_column][seq_length:train_size]
    fig.add_trace(go.Scatter(x=train_dates, y=train_predict.flatten(), mode='lines', name='Prédictions d\'entraînement', line=dict(color='green')))

    # Prédictions de test
    test_dates = df[date_column][train_size+seq_length:]
    fig.add_trace(go.Scatter(x=test_dates, y=test_predict.flatten(), mode='lines', name='Prédictions de test', line=dict(color='red')))

    # Prédictions futures
    last_date = df[date_column].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_predict))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predict.flatten(), mode='lines', name='Prédictions futures', line=dict(color='purple')))

    fig.update_layout(title="Prédictions d'exportation vs Réalité",
                      xaxis_title='Date',
                      yaxis_title='Exportations',
                      legend_title='Légende')
    return fig

# Application Streamlit
def main_excel():
    if 'step' not in st.session_state:
        st.session_state.step = 0
    
    st.title("Analyse de données et entraînement de modèle")
    
    # Étape 1 : Chargement du fichier
    uploaded_file = st.file_uploader("Choisissez un fichier Excel ou CSV", type=["xlsx", "csv"])
    
    if uploaded_file is not None or st.session_state.step>=1:
        st.session_state.step = max(st.session_state.step, 1)
        
        # Demander les noms des colonnes
        date_column = st.text_input("Nom de la colonne de date", "Date")
        colis_column = st.text_input("Nom de la colonne de nombre de colis", "nombre de colis")
        
        if st.button("Charger les données") or st.session_state.step>=2:
            st.session_state.step = max(st.session_state.step, 2)
            try:
                df = preprocess(uploaded_file, date_column, colis_column)
                st.session_state.df = df
                st.success("Données chargées avec succès!")
                
                # Afficher quelques informations sur les données
                st.write("Aperçu des données:")
                st.write(df.head())
                st.write(f"Nombre total d'enregistrements: {len(df)}")
                st.write(f"Période couverte: du {df[date_column].min()} au {df[date_column].max()}")
                
                # Afficher le graphique des données d'entrée
                st.plotly_chart(plot_input_data(df, date_column, colis_column))
                
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Erreur lors du chargement des données: {str(e)}")
    
    # Étape 2 : Entraînement du modèle
    if 'data_loaded' in st.session_state and st.session_state.data_loaded and uploaded_file:
        st.header("Entraînement du modèle")
        epochs = st.slider("Nombre d'époques", min_value=10, max_value=1000, value=400, step=10)
        
        if st.button("Entraîner le modèle"):
            with st.spinner("Entraînement du modèle en cours..."):
                df = st.session_state.df
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df[colis_column].values.reshape(-1, 1))
                
                seq_length = 30
                X, y = create_sequences(scaled_data, seq_length)
                
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                
                model, history = train_model(X_train, y_train, epochs)
                
                st.session_state.model = model
                st.session_state.history = history
                st.session_state.scaler = scaler
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.train_size = train_size
                st.session_state.seq_length = seq_length
                
                st.success("Modèle entraîné avec succès!")
    
    # Étape 3 : Affichage des résultats
    if 'model' in st.session_state:
        st.header("Résultats de l'entraînement")
        
        # Afficher la courbe de perte
        st.subheader("Courbes d'apprentissage")
        st.plotly_chart(plot_learning_curves(st.session_state.history))
        
        # Faire des prédictions
        train_predict = st.session_state.model.predict(X_train)
        test_predict = st.session_state.model.predict(st.session_state.X_test)
        train_predict = st.session_state.scaler.inverse_transform(train_predict)
        test_predict = st.session_state.scaler.inverse_transform(test_predict)
        
        # Prédictions futures
        last_sequence = scaled_data[-seq_length:]
        future_predictions = []
        for _ in range(30):  # Prédire pour les 30 prochains jours
            next_pred = st.session_state.model.predict(last_sequence.reshape(1, seq_length, 1))
            future_predictions.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred
        future_predictions = st.session_state.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Afficher les prédictions vs réalité
        st.subheader("Prédictions vs Réalité")
        st.plotly_chart(plot_predictions(
            st.session_state.df, 
            train_predict, 
            test_predict, 
            future_predictions, 
            st.session_state.train_size, 
            st.session_state.seq_length,
            date_column,
            colis_column
        ))

if __name__ == "__main__":
    main_excel()