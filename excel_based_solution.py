import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from neuralprophet import NeuralProphet
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(uploaded_file, date_column, colis_column, sheet_name=None):
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'xlsx':
        # Si sheet_name est None, pd.read_excel chargera la première feuille par défaut
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, index_col=None)
    elif file_type == 'csv':
        df = pd.read_csv(uploaded_file, index_col=None)
    else:
        raise ValueError("Format de fichier non pris en charge")
    
    if date_column not in df.columns or colis_column not in df.columns:
        raise ValueError(f"Le fichier doit contenir les colonnes '{date_column}' et '{colis_column}'")
    
    df[date_column] = pd.to_datetime(df[date_column])
    df[colis_column] = df[colis_column].fillna(0).astype(int)
    df_copy = df.rename(columns={date_column: "ds", colis_column: "y"})[["ds","y"]].copy()
    
    return df_copy

def plot_input_data(df):
    fig = px.line(df, x="ds", y="y", title="Données d'exportation")
    fig.update_layout(xaxis_title="Date", yaxis_title="Exportations")
    return fig

def plot_average_by_weekday(df):
    df_copy = df.copy()
    df_copy['Weekday'] = df_copy['ds'].dt.day_name()
    avg_by_weekday = df_copy.groupby('Weekday')['y'].mean()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_by_weekday = avg_by_weekday.reindex(weekday_order)
    
    fig = px.line(x=avg_by_weekday.index, y=avg_by_weekday.values, 
                  labels={'x': 'Jour de la semaine', 'y': 'Nombre moyen de colis'},
                  title='Demande moyenne par jour de la semaine')
    return fig

def plot_monthly_evolution(df):
    df_copy = df.copy()
    df_copy['YearMonth'] = df_copy['ds'].dt.to_period('M')
    monthly_avg = df_copy.groupby('YearMonth')['y'].mean().reset_index()
    monthly_avg['YearMonth'] = monthly_avg['YearMonth'].astype(str)
    
    fig = px.line(monthly_avg, x='YearMonth', y='y',
                  labels={'YearMonth': 'Mois', 'y': 'Nombre moyen de colis'},
                  title='Évolution mensuelle de la demande')
    fig.update_xaxes(tickangle=45)
    return fig

def plot_holiday_impact(forecast):
    holidays = ['Armistice', 'Ascension', 'Assomption', 'Fête de la Victoire', 'Fête du Travail', 
                'Fête nationale', "Jour de l'an", 'Lundi de Pentecôte', 'Lundi de Pâques', 
                'Noël', 'Toussaint']
    
    holiday_impact = forecast[['ds'] + [f'event_{h}' for h in holidays]].melt(id_vars=['ds'], 
                                                                              var_name='holiday', 
                                                                              value_name='impact')
    holiday_impact['holiday'] = holiday_impact['holiday'].str.replace('event_', '')
    
    fig = px.box(holiday_impact, x='holiday', y='impact', 
                 title="Impact des jours fériés sur la demande",
                 labels={'holiday': 'Jour férié', 'impact': 'Impact sur la demande'})
    fig.update_xaxes(tickangle=45)
    return fig

def plot_seasonal_trends(forecast):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Tendance annuelle", "Tendance hebdomadaire"))
    
    # Tendance annuelle
    yearly_trend = forecast.groupby(forecast['ds'].dt.dayofyear)['season_yearly'].mean().reset_index()
    fig.add_trace(go.Scatter(x=yearly_trend['ds'], y=yearly_trend['season_yearly'], 
                             mode='lines', name='Tendance annuelle'), row=1, col=1)
    
    # Tendance hebdomadaire
    weekly_trend = forecast.groupby(forecast['ds'].dt.dayofweek)['season_weekly'].mean().reset_index()
    fig.add_trace(go.Scatter(x=weekly_trend['ds'], y=weekly_trend['season_weekly'], 
                             mode='lines', name='Tendance hebdomadaire'), row=2, col=1)
    
    fig.update_layout(height=600, title_text="Décomposition des tendances saisonnières")
    return fig

def plot_residuals(df, forecast):
    residuals = df['y'] - forecast['yhat1']
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Résidus au fil du temps", "Distribution des résidus"))
    
    # Résidus au fil du temps
    fig.add_trace(go.Scatter(x=df['ds'], y=residuals, mode='markers', name='Résidus'), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Distribution des résidus
    fig.add_trace(go.Histogram(x=residuals, name='Distribution des résidus'), row=2, col=1)
    
    fig.update_layout(height=600, title_text="Analyse des résidus")
    return fig

def calculate_model_metrics(df, model):
    # Obtenir les prédictions sur les données d'entraînement
    train_predictions = model.predict(df)
    
    y_true = df['y']
    y_pred = train_predictions['yhat1'] if 'yhat1' in train_predictions.columns else train_predictions['yhat']
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mae, mse, r2

def plot_forecast(df, forecast, train_predictions):
    fig = go.Figure()
    
    # Données réelles
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Données réelles'))
    
    # Prédictions d'entraînement
    y_pred_train = 'yhat1' if 'yhat1' in train_predictions.columns else 'yhat'
    fig.add_trace(go.Scatter(x=train_predictions['ds'], y=train_predictions[y_pred_train], 
                             mode='lines', name='Prédictions d\'entraînement', line=dict(color='green')))
    
    # Prédictions futures
    y_pred_future = 'yhat1' if 'yhat1' in forecast.columns else 'yhat'
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast[y_pred_future], 
                             mode='lines', name='Prédictions futures', line=dict(color='red')))
    
    fig.update_layout(title="Prédictions vs Réalité", xaxis_title="Date", yaxis_title="Exportations")
    return fig

def initialize_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'forecast' not in st.session_state:
        st.session_state.forecast = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {
            'future_periods': 300,
            'epochs': 20,
            'n_lags': 0
        }
    if 'regularization' not in st.session_state:
        st.session_state.regularization = {
            'trend': 0.1,
            'seasonality': 0.1,
            'auto_regression': 0.1,
        }
    if 'model_components' not in st.session_state:
        st.session_state.model_components = {
            'trend': True,
            'seasonality': True,
            'auto_regression': False,
        }

def load_data_button_click():
    try:
        st.session_state.df = load_data(st.session_state.uploaded_file, 
                                        st.session_state.date_column, 
                                        st.session_state.colis_column)
        st.session_state.data_loaded = True
        st.session_state.model_trained = False  # Reset model state when new data is loaded
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        st.session_state.data_loaded = False

def train_model_button_click():
    if st.session_state.data_loaded:
        with st.spinner("Entraînement du modèle en cours..."):
            growth = "linear" if st.session_state.model_components['trend'] else "off"
            n_lags = st.session_state.model_params['n_lags'] if st.session_state.model_components['auto_regression'] else 0
            
            m = NeuralProphet(
                growth=growth,
                yearly_seasonality=st.session_state.model_components['seasonality'],
                weekly_seasonality=st.session_state.model_components['seasonality'],
                daily_seasonality=False,
                trend_reg=st.session_state.regularization['trend'],
                seasonality_reg=st.session_state.regularization['seasonality'],
                n_lags=n_lags,
                n_forecasts=1,
                ar_reg=st.session_state.regularization['auto_regression'] if st.session_state.model_components['auto_regression'] else None,
                epochs=st.session_state.model_params['epochs']
            )
            m = m.add_country_holidays("FR")
            metrics = m.fit(st.session_state.df, freq="D")

        with st.spinner("Génération des prédictions..."):
            # Prédictions sur les données d'entraînement
            st.session_state.train_predictions = m.predict(st.session_state.df)
            
            # Prédictions futures
            future = m.make_future_dataframe(st.session_state.df, 
                                             periods=st.session_state.model_params['future_periods'])
            st.session_state.forecast = m.predict(future)
            y_pred_future = 'yhat1' if 'yhat1' in st.session_state.forecast.columns else 'yhat'
            st.session_state.forecast[y_pred_future] = st.session_state.forecast[y_pred_future].apply(lambda x:max(x,0))
        
        st.session_state.model_trained = True
        st.session_state.model = m
        st.success("Modèle entraîné et prédictions générées avec succès!")
    else:
        st.error("Veuillez d'abord charger les données.")

def main_excel():
    st.title("Analyse de données et prédiction des exportations")
    
    initialize_session_state()
    
    st.session_state.uploaded_file = st.file_uploader("Choisissez un fichier Excel ou CSV", type=["xlsx", "csv"])
    
    if st.session_state.uploaded_file is not None:
        file_type = st.session_state.uploaded_file.name.split('.')[-1].lower()
        
        st.session_state.date_column = st.text_input("Nom de la colonne de date", "Date")
        st.session_state.colis_column = st.text_input("Nom de la colonne de nombre de colis", "nombre de colis")
        
        if file_type == 'xlsx':
            # Obtenir la liste des noms de feuilles
            xls = pd.ExcelFile(st.session_state.uploaded_file)
            sheet_names = xls.sheet_names
            
            # Ajouter un sélecteur de feuille
            selected_sheet = st.selectbox("Choisissez une feuille (optionnel)", [""] + sheet_names)
            
            # Si aucune feuille n'est sélectionnée, on passe None à la fonction load_data
            sheet_name = selected_sheet if selected_sheet != "" else None
        else:
            sheet_name = None
        
        if st.button("Charger les données"):
            try:
                st.session_state.df = load_data(st.session_state.uploaded_file, 
                                                st.session_state.date_column, 
                                                st.session_state.colis_column,
                                                sheet_name)
                st.session_state.data_loaded = True
                st.session_state.model_trained = False  # Reset model state when new data is loaded
                st.success("Données chargées et préparées avec succès!")
            except Exception as e:
                st.error(f"Erreur lors du chargement des données : {str(e)}")
                st.session_state.data_loaded = False
        
        
        if st.session_state.data_loaded:            
            st.subheader("Aperçu des données")
            st.write(st.session_state.df.head())
            st.write(f"Nombre total d'enregistrements: {len(st.session_state.df)}")
            st.write(f"Période couverte: du {st.session_state.df['ds'].min()} au {st.session_state.df['ds'].max()}")
            
            st.plotly_chart(plot_input_data(st.session_state.df))
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_average_by_weekday(st.session_state.df))
            with col2:
                st.plotly_chart(plot_monthly_evolution(st.session_state.df))
            
            st.subheader("Configuration du modèle NeuralProphet")
            st.session_state.model_components['trend'] = st.checkbox("Inclure la tendance", value=True)
            st.session_state.model_components['seasonality'] = st.checkbox("Inclure la saisonnalité", value=True)
            
            st.session_state.model_params['future_periods'] = st.slider("Nombre de jours à prédire", 
                                                                        min_value=10, max_value=1000, 
                                                                        value=st.session_state.model_params['future_periods'])
            st.session_state.model_params['epochs'] = st.slider("Nombre d'époques", 
                                                                min_value=10, max_value=300, 
                                                                value=st.session_state.model_params['epochs'])

            if st.button("Entraîner le modèle et faire des prédictions"):
                train_model_button_click()
            
            if st.session_state.model_trained:
                try:
                    forecast_plot = plot_forecast(st.session_state.df, st.session_state.forecast, st.session_state.train_predictions)
                    st.plotly_chart(forecast_plot)
                    st.plotly_chart(plot_holiday_impact(st.session_state.forecast))
                    st.plotly_chart(plot_seasonal_trends(st.session_state.forecast))
                    st.plotly_chart(plot_residuals(st.session_state.df, st.session_state.train_predictions))
                    mae, mse, r2 = calculate_model_metrics(st.session_state.df, st.session_state.model)
                    st.subheader("Métriques de performance du modèle (sur les données d'entraînement)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{mae:.2f}")
                    col2.metric("MSE", f"{mse:.2f}")
                    col3.metric("R²", f"{r2:.2f}")
                except Exception as e:
                    st.error(f"Erreur lors de la création des graphiques : {str(e)}")

if __name__ == "__main__":
    main_excel()