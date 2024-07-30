import streamlit as st
import pandas as pd
from export_analyzer import ExportAnalyzer

def initialize_session_state():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ExportAnalyzer()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

def main_excel():
    st.title("Analyse de données et prédiction des exportations")
    
    initialize_session_state()
    
    uploaded_file = st.file_uploader("Choisissez un fichier Excel ou CSV", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        date_column = st.text_input("Nom de la colonne de date", "Date")
        colis_column = st.text_input("Nom de la colonne de nombre de colis", "nombre de colis")
        
        sheet_name = None
        if file_type == 'xlsx':
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            selected_sheet = st.selectbox("Choisissez une feuille (optionnel)", ["Première feuille"] + sheet_names)
            sheet_name = None if selected_sheet == "Première feuille" else selected_sheet
        
        if st.button("Charger les données"):
            if st.session_state.analyzer.load_data(uploaded_file, date_column, colis_column, sheet_name):
                st.session_state.data_loaded = True
                st.success("Données chargées et préparées avec succès!")
        
        if st.session_state.data_loaded:            
            st.subheader("Aperçu des données")
            st.write(st.session_state.analyzer.df.head())
            st.write(f"Nombre total d'enregistrements: {len(st.session_state.analyzer.df)}")
            st.write(f"Période couverte: du {st.session_state.analyzer.df['ds'].min()} au {st.session_state.analyzer.df['ds'].max()}")
            
            st.plotly_chart(st.session_state.analyzer.plot_input_data())
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(st.session_state.analyzer.plot_average_by_weekday())
            with col2:
                st.plotly_chart(st.session_state.analyzer.plot_monthly_evolution())
            
            st.subheader("Configuration du modèle NeuralProphet")
            st.session_state.analyzer.model_components['trend'] = st.checkbox("Inclure la tendance", value=True)
            st.session_state.analyzer.model_components['seasonality'] = st.checkbox("Inclure la saisonnalité", value=True)
            
            st.session_state.analyzer.model_params['future_periods'] = st.slider("Nombre de jours à prédire", 
                                                                min_value=10, max_value=1000, 
                                                                value=st.session_state.analyzer.model_params['future_periods'])
            st.session_state.analyzer.model_params['epochs'] = st.slider("Nombre d'époques", 
                                                        min_value=10, max_value=300, 
                                                        value=st.session_state.analyzer.model_params['epochs'])

            if st.button("Entraîner le modèle et faire des prédictions"):
                with st.spinner("Entraînement du modèle en cours..."):
                    metrics = st.session_state.analyzer.train_model()
                st.session_state.model_trained = True
                st.success("Modèle entraîné et prédictions générées avec succès!")
            
        if st.session_state.model_trained:
            try:
                st.plotly_chart(st.session_state.analyzer.plot_forecast())
                st.plotly_chart(st.session_state.analyzer.plot_holiday_impact())
                st.plotly_chart(st.session_state.analyzer.plot_seasonal_trends())
                st.plotly_chart(st.session_state.analyzer.plot_residuals())
                mae, mse, r2 = st.session_state.analyzer.calculate_model_metrics()
                st.subheader("Métriques de performance du modèle (sur les données d'entraînement)")
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("MSE", f"{mse:.2f}")
                col3.metric("R²", f"{r2:.2f}")
            except Exception as e:
                st.error(f"Erreur lors de la création des graphiques : {str(e)}")

if __name__ == "__main__":
    main_excel()