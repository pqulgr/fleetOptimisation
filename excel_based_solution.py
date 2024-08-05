import streamlit as st
import pandas as pd
from export_analyzer import ExportAnalyzer
from fleet_estimator import FleetEstimator

def initialize_session_state():
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            'analyzer': ExportAnalyzer(),
            'data_loaded': False,
            'model_trained': False,
            'fleet_estimated': False,
            'fleet_estimator': None,
            'fleet_size': None,
            'cost_option': None,
            'cost_params': None
        }

def get_cost_options():
    cost_params = {}
    cost_option = st.selectbox("Sélectionnez une option de coût", ("Coût basé sur les points de collecte", "Coût basé sur le poids"))
    cost_option = "Option 1" if cost_option == "Coût basé sur les points de collecte" else "Option 2"
    
    if cost_option == "Option 1":
        st.write("Délai avant reverse (jours)")
        col1, col2 = st.columns(2)
        with col1:
            min_reverse_time = st.number_input("Min", min_value=1, max_value=20, value=1, step=1)
        with col2:
            max_reverse_time = st.number_input("Max", min_value=1, max_value=20, value=3, step=1)
        if min_reverse_time > max_reverse_time:
            min_reverse_time, max_reverse_time = max_reverse_time, min_reverse_time
        cost_params["reverse_time"] = (min_reverse_time, max_reverse_time)
        cost_params["nb_locations"] = st.number_input("Nombre de destinations", min_value=1, step=1, value=1)
        cost_params["cost_emballage"] = st.number_input("Coût d'achat des emballages", min_value=0.0, step=0.01, value=2.0)
        cost_params["cost_location"] = st.number_input("Coût de récupération à un point relai", min_value=0.0, step=0.1, value=5.0)
        cost_params["cost_per_demand"] = st.number_input("Coût par envoi", min_value=0.0, step=0.1, value=1.0)
    elif cost_option == "Option 2":
        cost_params["reverse_time"] = st.number_input("Délai avant reverse (jours)", min_value=1, step=1, value=3)
        cost_params["cost_emballage"] = st.number_input("Coût d'achat des emballages", min_value=0.0, step=0.01, value=2.0)
        cost_params["poids_sac"] = st.number_input("Poids moyen", min_value=0.0, step=0.1, value=1.0)
        cost_params["cost_per_demand"] = st.number_input("Coût par envoi par Kg", min_value=0.0, step=0.1, value=1.0)
        cost_params["cost_per_return"] = st.number_input("Coût par retour par Kg", min_value=0.0, step=0.1, value=1.0)

    return cost_option, cost_params

def display_data_analysis():
    st.subheader("Aperçu des données")
    st.table(st.session_state.app_state['analyzer'].df_original.head())
    st.write(f"Nombre total d'enregistrements: {len(st.session_state.app_state['analyzer'].df)}")
    st.write(f"Période couverte: du {st.session_state.app_state['analyzer'].df['ds'].min()} au {st.session_state.app_state['analyzer'].df['ds'].max()}")
    
    st.plotly_chart(st.session_state.app_state['analyzer'].plot_input_data())
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(st.session_state.app_state['analyzer'].plot_average_by_weekday())
    with col2:
        st.plotly_chart(st.session_state.app_state['analyzer'].plot_monthly_evolution())

def display_model_results():
    try:
        st.session_state.app_state['analyzer'].display_model_summary()
        st.session_state.app_state['analyzer'].display_model_metrics()
        st.plotly_chart(st.session_state.app_state['analyzer'].plot_forecast())
        st.plotly_chart(st.session_state.app_state['analyzer'].plot_holiday_impact())
        st.plotly_chart(st.session_state.app_state['analyzer'].plot_seasonal_trends())
        st.plotly_chart(st.session_state.app_state['analyzer'].plot_component_importance())
        st.plotly_chart(st.session_state.app_state['analyzer'].plot_forecast_components())
        st.plotly_chart(st.session_state.app_state['analyzer'].plot_residuals())
        
    except Exception as e:
        st.error(f"Erreur lors de la création des graphiques : {str(e)}")

def estimate_fleet_size():
    with st.spinner("Estimation de la taille de la flotte en cours..."):
        true_data = st.session_state.app_state['analyzer'].df
        predictions = st.session_state.app_state['analyzer'].forecast
        
        st.session_state.app_state['fleet_estimator'] = FleetEstimator(
            true_data, 
            predictions, 
            st.session_state.app_state['cost_option'], 
            st.session_state.app_state['cost_params']
        )
        st.session_state.app_state['fleet_size'] = st.session_state.app_state['fleet_estimator'].estimate_fleet_size()
        st.session_state.app_state['fleet_estimated'] = True

def display_fleet_estimation():
    st.session_state.app_state['fleet_estimator'].display_results()

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
            if st.session_state.app_state['analyzer'].load_data(uploaded_file, date_column, colis_column, sheet_name):
                st.session_state.app_state['data_loaded'] = True
                st.success("Données chargées et préparées avec succès!")
            else:
                st.error("Erreur lors du chargement des données.")
        
        if st.session_state.app_state['data_loaded']:
            display_data_analysis()
            
            st.subheader("Configuration du modèle NeuralProphet")
            st.session_state.app_state['analyzer'].model_components['trend'] = st.checkbox("Inclure la tendance", value=True)
            st.session_state.app_state['analyzer'].model_components['artificial_noise'] = st.checkbox("Inclure du bruit", value=True)

            st.session_state.app_state['analyzer'].model_params['future_periods'] = st.slider("Nombre de jours à prédire", 
                                                                min_value=10, max_value=1000, 
                                                                value=st.session_state.app_state['analyzer'].model_params['future_periods'])
            st.session_state.app_state['analyzer'].model_params['epochs'] = st.slider("Nombre d'époques", 
                                                        min_value=10, max_value=300, 
                                                        value=st.session_state.app_state['analyzer'].model_params['epochs'])
            
            st.session_state.app_state['analyzer'].use_country_holidays = st.session_state.app_state['analyzer'].get_event_features()

            if st.button("Entraîner le modèle et faire des prédictions"):
                with st.spinner("Entraînement du modèle en cours..."):
                    st.session_state.app_state['analyzer'].train_model()
                st.session_state.app_state['model_trained'] = True
                st.success("Modèle entraîné et prédictions générées avec succès!")
            
        if st.session_state.app_state['model_trained']:
            display_model_results()

            st.subheader("Estimation de la taille de la flotte")
            st.session_state.app_state['cost_option'], st.session_state.app_state['cost_params'] = get_cost_options()

            if st.button("Estimer la taille de la flotte"):
                estimate_fleet_size()

        if st.session_state.app_state['fleet_estimated']:
            display_fleet_estimation()
    else:
        initialize_session_state()

if __name__ == "__main__":
    main_excel()