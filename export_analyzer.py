import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class ExportAnalyzer:
    def __init__(self):
        self.df = None
        self.model = None
        self.forecast = None
        self.train_predictions = None
        self.data_loaded = False
        self.model_trained = False
        self.model_params = {
            'future_periods': 300,
            'epochs': 20,
        }
        self.model_components = {
            'trend': True,
            'seasonality': True,
            'auto_regression': False,
            'artificial_noise':False
        }

    def load_data(self, uploaded_file, date_column, colis_column, sheet_name=None):
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == 'xlsx':
                if sheet_name is None:
                    # Si aucune feuille n'est sélectionnée, on lit la première feuille
                    df = pd.read_excel(uploaded_file, sheet_name=0, index_col=None)
                else:
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, index_col=None)
            elif file_type == 'csv':
                df = pd.read_csv(uploaded_file, index_col=None)
            else:
                raise ValueError("Format de fichier non pris en charge")
            
            if date_column not in df.columns or colis_column not in df.columns:
                raise ValueError(f"Le fichier doit contenir les colonnes '{date_column}' et '{colis_column}'")
            
            df[date_column] = pd.to_datetime(df[date_column])
            df[colis_column] = df[colis_column].fillna(0).astype(int)
            self.df = df.rename(columns={date_column: "ds", colis_column: "y"})[["ds","y"]].copy()
            self.data_loaded = True
            self.model_trained = False
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {str(e)}")
            self.data_loaded = False
            return False

    def plot_input_data(self):
        fig = px.line(self.df, x="ds", y="y", title="Données d'exportation")
        fig.update_layout(xaxis_title="Date", yaxis_title="Exportations")
        return fig

    def plot_average_by_weekday(self):
        df_copy = self.df.copy()
        df_copy['Weekday'] = df_copy['ds'].dt.day_name()
        avg_by_weekday = df_copy.groupby('Weekday')['y'].mean()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        avg_by_weekday = avg_by_weekday.reindex(weekday_order)
        
        fig = px.line(x=avg_by_weekday.index, y=avg_by_weekday.values, 
                      labels={'x': 'Jour de la semaine', 'y': 'Nombre moyen de colis'},
                      title='Demande moyenne par jour de la semaine')
        return fig

    def plot_monthly_evolution(self):
        df_copy = self.df.copy()
        df_copy['YearMonth'] = df_copy['ds'].dt.to_period('M')
        monthly_avg = df_copy.groupby('YearMonth')['y'].mean().reset_index()
        monthly_avg['YearMonth'] = monthly_avg['YearMonth'].astype(str)
        
        fig = px.line(monthly_avg, x='YearMonth', y='y',
                      labels={'YearMonth': 'Mois', 'y': 'Nombre moyen de colis'},
                      title='Évolution mensuelle de la demande')
        fig.update_xaxes(tickangle=45)
        return fig
    
    def add_noise(self, y_predictions, y_pred_future):
        # Calculer la différence entre les données réelles et les prédictions sur les données d'entraînement
        y_true = self.df['y']
        y_pred_train = self.train_predictions[y_pred_future]
        
        residuals = y_true - y_pred_train
        
        # Calculer la moyenne et l'écart-type des résidus
        #mean_residual = np.mean(residuals)
        #std_residual = np.std(residuals)
        
        # Générer du bruit basé sur la distribution des résidus
        noise = np.random.choice(residuals, size=len(y_predictions))
        
        # Ajouter le bruit aux prédictions futures
        noisy_predictions = np.array([y_predictions[i] + noise[i] if y_predictions[i]>0 else y_predictions[i] for i in range(len(noise))])
        
        return noisy_predictions

    def train_model(self):
        growth = "linear" if self.model_components['trend'] else "off"
        
        self.model = NeuralProphet(
            growth=growth,
            yearly_seasonality=self.model_components['seasonality'],
            weekly_seasonality=self.model_components['seasonality'],
            daily_seasonality=False,
            n_forecasts=1,
            epochs=self.model_params['epochs']
        )
        self.model = self.model.add_country_holidays("FR")
        metrics = self.model.fit(self.df, freq="D")
        
        self.train_predictions = self.model.predict(self.df)
        
        future = self.model.make_future_dataframe(self.df, 
                                                  periods=self.model_params['future_periods'])
        self.forecast = self.model.predict(future)
        y_pred_future = 'yhat1' if 'yhat1' in self.forecast.columns else 'yhat'
        if self.model_components['artificial_noise']:
            self.forecast[y_pred_future] = self.add_noise(self.forecast[y_pred_future], y_pred_future)
        self.forecast[y_pred_future] = self.forecast[y_pred_future].apply(lambda x: max(x, 0))
        
        self.model_trained = True
        return metrics

    def plot_forecast(self):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=self.df['ds'], y=self.df['y'], mode='lines', name='Données réelles'))
        
        y_pred_train = 'yhat1' if 'yhat1' in self.train_predictions.columns else 'yhat'
        fig.add_trace(go.Scatter(x=self.train_predictions['ds'], y=self.train_predictions[y_pred_train], 
                                 mode='lines', name="Prédictions d'entraînement", line=dict(color='green')))
        
        y_pred_future = 'yhat1' if 'yhat1' in self.forecast.columns else 'yhat'
        fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast[y_pred_future], 
                                 mode='lines', name='Prédictions futures', line=dict(color='red')))
        
        fig.update_layout(title="Prédictions vs Réalité", xaxis_title="Date", yaxis_title="Exportations")
        return fig

    def plot_holiday_impact(self):
        holidays = ['Armistice', 'Ascension', 'Assomption', 'Fête de la Victoire', 'Fête du Travail', 
                    'Fête nationale', "Jour de l'an", 'Lundi de Pentecôte', 'Lundi de Pâques', 
                    'Noël', 'Toussaint']
        
        holiday_impact = self.forecast[['ds'] + [f'event_{h}' for h in holidays]].melt(id_vars=['ds'], 
                                                                                  var_name='holiday', 
                                                                                  value_name='impact')
        holiday_impact['holiday'] = holiday_impact['holiday'].str.replace('event_', '')
        
        fig = px.box(holiday_impact, x='holiday', y='impact', 
                     title="Impact des jours fériés sur la demande",
                     labels={'holiday': 'Jour férié', 'impact': 'Impact sur la demande'})
        fig.update_xaxes(tickangle=45)
        return fig

    def plot_seasonal_trends(self):
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Tendance annuelle", "Tendance hebdomadaire"))
        
        yearly_trend = self.forecast.groupby(self.forecast['ds'].dt.dayofyear)['season_yearly'].mean().reset_index()
        fig.add_trace(go.Scatter(x=yearly_trend['ds'], y=yearly_trend['season_yearly'], 
                                 mode='lines', name='Tendance annuelle'), row=1, col=1)
        
        weekly_trend = self.forecast.groupby(self.forecast['ds'].dt.dayofweek)['season_weekly'].mean().reset_index()
        fig.add_trace(go.Scatter(x=weekly_trend['ds'], y=weekly_trend['season_weekly'], 
                                 mode='lines', name='Tendance hebdomadaire'), row=2, col=1)
        
        fig.update_layout(height=600, title_text="Décomposition des tendances saisonnières")
        return fig

    def plot_residuals(self):
        residuals = self.df['y'] - self.train_predictions['yhat1']
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Résidus au fil du temps", "Distribution des résidus"))
        
        fig.add_trace(go.Scatter(x=self.df['ds'], y=residuals, mode='markers', name='Résidus'), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        fig.add_trace(go.Histogram(x=residuals, name='Distribution des résidus'), row=2, col=1)
        
        fig.update_layout(height=600, title_text="Analyse des résidus")
        return fig

    def calculate_model_metrics(self):
        y_true = self.df['y']
        y_pred = self.train_predictions['yhat1'] if 'yhat1' in self.train_predictions.columns else self.train_predictions['yhat']
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return mae, mse, r2