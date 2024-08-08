import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
from scipy import stats

class ExportAnalyzer:
    def __init__(self):
        self.df = None
        self.model = None
        self.forecast = None
        self.train_predictions = None
        self.use_country_holidays = False
        self.df_original = None 
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
            'artificial_noise':False,
            'freq':'D'
        }

    def get_event_features(self):
        st.subheader("Ajouter des event features")
        
        use_country_holidays = st.checkbox("Utiliser les jours fériés français", value=True)

        return use_country_holidays

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
            self.df_original = df.copy()
            
            # Créer un DataFrame spécifique pour NeuralProphet avec seulement les colonnes nécessaires
            self.df = df.rename(columns={date_column: "ds", colis_column: "y"})[["ds", "y"]].copy()
            
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

    def calculate_model_metrics(self):
        y_true = self.df['y']
        y_pred = self.train_predictions['yhat1'] if 'yhat1' in self.train_predictions.columns else self.train_predictions['yhat']
        
        # Métriques de base
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Coefficient de Theil U
        y_true_shift = y_true[1:]
        y_true = y_true[:-1]
        y_pred = y_pred[:-1]
        numerator = np.sqrt(np.mean((y_true - y_pred)**2))
        denominator = np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2))
        theil_u = numerator / denominator
        
        # Décomposition de l'erreur
        error = y_true - y_pred
        bias = np.mean(error)
        variance = np.var(error)
        
        # Test de Durbin-Watson pour l'autocorrélation des résidus
        dw_statistic = np.sum(np.diff(error)**2) / np.sum(error**2)
        
        # Normalité des résidus (test de Shapiro-Wilk)
        _, p_value = stats.shapiro(error)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Theil U': theil_u,
            'Biais': bias,
            'Variance de l erreur': variance,
            'Statistique de Durbin-Watson': dw_statistic,
            'P-value du test de Shapiro-Wilk': p_value
        }
        
        return metrics

    def display_model_metrics(self):
        metrics = self.calculate_model_metrics()
        
        st.subheader("Métriques de performance du modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("MAE (Erreur Absolue Moyenne)", f"{metrics['MAE']:.2f}")
            st.metric("MSE (Erreur Quadratique Moyenne)", f"{metrics['MSE']:.2f}")
            st.metric("RMSE (Racine de l'Erreur Quadratique Moyenne)", f"{metrics['RMSE']:.2f}")
            st.metric("R² (Coefficient de détermination)", f"{metrics['R²']:.2f}")
        
        with col2:
            st.metric("Coefficient de Theil U", f"{metrics['Theil U']:.2f}")
            st.metric("Biais", f"{metrics['Biais']:.2f}")
            st.metric("Variance de l'erreur", f"{metrics['Variance de l erreur']:.2f}")
            st.metric("Statistique de Durbin-Watson", f"{metrics['Statistique de Durbin-Watson']:.2f}")
        
        st.write(f"P-value du test de Shapiro-Wilk : {metrics['P-value du test de Shapiro-Wilk']:.4f}")
        
        st.write("Interprétation des métriques :")
        st.write("- MAE, MSE, RMSE : Plus ces valeurs sont basses, meilleur est le modèle.")
        st.write("- R² : Plus proche de 1, meilleur est le modèle.")
        st.write("- Theil U : Varie entre 0 et 1. Plus proche de 0, meilleur est le modèle.")
        st.write("- Statistique de Durbin-Watson : Proche de 2 indique une absence d'autocorrélation des résidus.")
        st.write("- Test de Shapiro-Wilk : Une p-value > 0.05 suggère que les résidus sont normalement distribués.")

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
        
        noise = np.random.choice(residuals, size=len(y_predictions))
        
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

        # Créer un nouveau DataFrame avec seulement les colonnes nécessaires
        df_train = self.df[['ds', 'y']].copy()

        if self.use_country_holidays:
            self.model = self.model.add_country_holidays("FR")

        metrics = self.model.fit(df_train, freq=self.model_components["freq"])
        
        self.train_predictions = self.model.predict(df_train)
        
        future = self.model.make_future_dataframe(df_train, 
                                                  periods=self.model_params['future_periods'])
        
        self.forecast = self.model.predict(future)
        y_pred_future = 'yhat1' if 'yhat1' in self.forecast.columns else 'yhat'
        if self.model_components['artificial_noise']:
            self.forecast[y_pred_future] = self.add_noise(self.forecast[y_pred_future], y_pred_future)
        self.forecast[y_pred_future] = self.forecast[y_pred_future].apply(lambda x: max(x, 0))
        
        self.model_trained = True
        return metrics


    def plot_residuals(self):
        residuals = self.df['y'] - self.train_predictions['yhat1']
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Résidus au fil du temps", "Distribution des résidus"))
        
        fig.add_trace(go.Scatter(x=self.df['ds'], y=residuals, mode='markers', name='Résidus'), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        fig.add_trace(go.Histogram(x=residuals, name='Distribution des résidus'), row=2, col=1)
        
        fig.update_layout(height=600, title_text="Analyse des résidus")
        return fig


    def plot_holiday_impact(self):
        # Liste des jours fériés français
        french_holidays = [
            'Armistice', 'Ascension', 'Assomption', 'Fête de la Victoire', 'Fête du Travail',
            'Fête nationale', "Jour de l'an", 'Lundi de Pentecôte', 'Lundi de Pâques',
            'Noël', 'Toussaint'
        ]
        
        # Combiner les jours fériés français et les événements personnalisés
        all_events = french_holidays
        
        # Filtrer pour n'inclure que les événements présents dans les prédictions
        events_in_forecast = [h for h in all_events if f'event_{h}' in self.forecast.columns]
        
        # Créer le DataFrame d'impact des événements
        holiday_impact = self.forecast[['ds'] + [f'event_{h}' for h in events_in_forecast]].melt(
            id_vars=['ds'],
            var_name='holiday',
            value_name='impact'
        )
        holiday_impact['holiday'] = holiday_impact['holiday'].str.replace('event_', '')
        
        # Calculer l'impact moyen pour chaque jour férié
        avg_impact = holiday_impact.groupby('holiday')['impact'].mean().sort_values(ascending=False)
        
        # Créer le graphique
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Impact moyen des jours fériés", "Distribution de l'impact par jour férié"),
                            vertical_spacing=0.3)

        # Boîte à moustaches pour la distribution de l'impact
        fig.add_trace(go.Box(x=holiday_impact['holiday'], y=holiday_impact['impact'], name="Distribution de l'impact"),
                      row=2, col=1)

        # Graphique à barres pour l'impact moyen
        fig.add_trace(go.Bar(x=avg_impact.index, y=avg_impact.values, name="Impact moyen"),
                      row=1, col=1)
        
        fig.update_layout(height=800, title_text="Analyse de l'impact des jours fériés sur la demande")
        fig.update_xaxes(tickangle=45, row=2, col=1)
        
        return fig

    def plot_forecast(self):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Prédictions vs Réalité", "Erreur de prédiction"))
        
        # Données réelles
        fig.add_trace(go.Scatter(x=self.df['ds'], y=self.df['y'], mode='lines', name='Données réelles'),
                      row=1, col=1)
        
        # Prédictions d'entraînement
        y_pred_train = 'yhat1' if 'yhat1' in self.train_predictions.columns else 'yhat'
        fig.add_trace(go.Scatter(x=self.train_predictions['ds'], y=self.train_predictions[y_pred_train], 
                                 mode='lines', name="Prédictions d'entraînement", line=dict(color='green')),
                      row=1, col=1)
        
        # Prédictions futures
        y_pred_future = 'yhat1' if 'yhat1' in self.forecast.columns else 'yhat'
        fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast[y_pred_future], 
                                 mode='lines', name='Prédictions futures', line=dict(color='red')),
                      row=1, col=1)
        
        # Erreur de prédiction
        error = self.df['y'] - self.train_predictions[y_pred_train]
        fig.add_trace(go.Scatter(x=self.df['ds'], y=error, mode='lines', name='Erreur de prédiction'),
                      row=2, col=1)        
        fig.update_layout(height=800, title_text="Analyse des prédictions du modèle")
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Exportations", row=1, col=1)
        fig.update_yaxes(title_text="Erreur", row=2, col=1)
        
        return fig

    def plot_seasonal_trends(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Tendance annuelle", "Tendance hebdomadaire", 
                                                            "Composante de tendance", "Résidus"))
        
        # Tendance annuelle
        yearly_trend = self.forecast.groupby(self.forecast['ds'].dt.dayofyear)['season_yearly'].mean().reset_index()
        fig.add_trace(go.Scatter(x=yearly_trend['ds'], y=yearly_trend['season_yearly'], 
                                 mode='lines', name='Tendance annuelle'), row=1, col=1)
        
        # Tendance hebdomadaire
        weekly_trend = self.forecast.groupby(self.forecast['ds'].dt.dayofweek)['season_weekly'].mean().reset_index()
        fig.add_trace(go.Scatter(x=weekly_trend['ds'], y=weekly_trend['season_weekly'], 
                                 mode='lines', name='Tendance hebdomadaire'), row=1, col=2)
        
        # Composante de tendance
        fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast['trend'], 
                                 mode='lines', name='Tendance générale'), row=2, col=1)
        
        # Résidus
        residuals = self.df['y'] - self.train_predictions['yhat1']
        fig.add_trace(go.Scatter(x=self.df['ds'], y=residuals, mode='markers', name='Résidus'), row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(height=800, title_text="Décomposition des composantes du modèle")
        return fig

    def plot_component_importance(self):
        components = ['trend', 'season_yearly', 'season_weekly']
        if self.use_country_holidays:
            components.extend([col for col in self.forecast.columns if col.startswith('event_')])
        
        importance = {}
        for component in components:
            if component in self.forecast.columns:
                importance[component] = np.abs(self.forecast[component]).mean()
        
        importance = pd.Series(importance).sort_values(ascending=True)
        
        fig = go.Figure(go.Bar(
            x=importance.values,
            y=importance.index,
            orientation='h'
        ))
        
        fig.update_layout(
            title="Importance relative des composantes du modèle",
            xaxis_title="Importance moyenne (valeur absolue)",
            yaxis_title="Composante",
            height=400 + len(importance) * 20
        )
        
        return fig

    def display_model_summary(self):
        INV_OPTION_FREQ = {
            "D": "Jours",
            "W": "Semaines",
            "MS": "Mois",
            "QS": "Trimestres"
        }
        st.subheader("Résumé du modèle")
        st.write("Paramètres du modèle:")
        st.write(f"- Périodes futures: {self.model_params['future_periods']}")
        st.write(f"- Époques: {self.model_params['epochs']}")
        st.write(f"- Tendance: {'Activée' if self.model_components['trend'] else 'Désactivée'}")
        st.write(f"- Saisonnalité: {'Activée' if self.model_components['seasonality'] else 'Désactivée'}")
        st.write(f"- Base temporelle: {INV_OPTION_FREQ[self.model_components['freq']]}")
        #st.write(f"- Autorégression: {'Activée' if self.model_components['auto_regression'] else 'Désactivée'}")
        st.write(f"- Bruit artificiel: {'Activé' if self.model_components['artificial_noise'] else 'Désactivé'}")
        st.write(f"- Jours fériés français: {'Inclus' if self.use_country_holidays else 'Non inclus'}")
    
    def plot_forecast_components(self):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Prédiction totale", "Tendance", "Saisonnalité"))
        
        # Prédiction totale
        fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast['yhat1'], 
                                 mode='lines', name='Prédiction totale'), row=1, col=1)
        
        # Tendance
        fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast['trend'], 
                                 mode='lines', name='Tendance'), row=2, col=1)
        
        # Saisonnalité (combinaison de la saisonnalité annuelle et hebdomadaire)
        seasonality = self.forecast['season_yearly'] + self.forecast['season_weekly']
        fig.add_trace(go.Scatter(x=self.forecast['ds'], y=seasonality, 
                                 mode='lines', name='Saisonnalité'), row=3, col=1)
        
        fig.update_layout(height=900, title_text="Décomposition des composantes de la prédiction")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Valeur", row=1, col=1)
        fig.update_yaxes(title_text="Valeur", row=2, col=1)
        fig.update_yaxes(title_text="Valeur", row=3, col=1)
        
        return fig