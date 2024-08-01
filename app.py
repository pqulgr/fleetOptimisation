import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (
    simulate_, plot_cdf, plot_3_scenarios, plot_return_probs, plot_shipping_demands,
    specific_fonction_for_accurately_determine_the_cost_of_the_recommended_number_of_bags,
    plot_cost_vs_reverse, create_summary_table, plot_reverse_optimal_fleet, run_simulation
)
from documentation import show_documentation
from excel_based_solution import main_excel

def initialiser_session_state():
    default_values = {
        'graphique_distribution': None,
        'table_stats': None,
        'mu_client': None,
        'sigma_client': None,
        'p_value': None
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_data(uploaded_file, date_column, colis_column, sheet_name=None):
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'xlsx':
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name or 0, index_col=None)
        elif file_type == 'csv':
            df = pd.read_csv(uploaded_file, index_col=None)
        else:
            raise ValueError("Format de fichier non pris en charge")

        if date_column not in df.columns or colis_column not in df.columns:
            raise ValueError(f"Le fichier doit contenir les colonnes '{date_column}' et '{colis_column}'")

        df[date_column] = pd.to_datetime(df[date_column])
        df[colis_column] = df[colis_column].fillna(0).astype(int)
        
        return df.rename(columns={date_column: "ds", colis_column: "y"})[["ds","y"]].copy()
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

def visualize_data(df):
    if df is None:
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogramme et densité des demandes d'export", "Q-Q Plot (Distribution Normale)"))

    # Histogram with KDE
    hist_values, bin_edges = np.histogram(df['y'], bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    kde = stats.gaussian_kde(df['y'])
    
    fig.add_trace(go.Bar(x=bin_centers, y=hist_values, name='Histogram'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bin_centers, y=kde(bin_centers), mode='lines', name='KDE'), row=1, col=1)

    mean, median = df['y'].mean(), df['y'].median()
    fig.add_vline(x=mean, line_dash="dash", line_color="red", annotation_text=f"Moyenne: {mean:.2f}", row=1, col=1)
    fig.add_vline(x=median, line_dash="dash", line_color="green", annotation_text=f"Médiane: {median:.2f}", row=1, col=1)

    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(df['y'], dist="norm", fit=True)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data'), row=1, col=2)
    fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', name='Fit'), row=1, col=2)

    fig.add_annotation(text=f'R² = {r**2:.4f}', xref="x2", yref="y2", x=0.05, y=0.95, showarrow=False)

    fig.update_layout(height=600, width=1000, title_text="Analyse de la distribution des demandes d'export")
    fig.update_xaxes(title_text="Nombre d'exports", row=1, col=1)
    fig.update_yaxes(title_text="Fréquence", row=1, col=1)
    fig.update_xaxes(title_text="Quantiles théoriques", row=1, col=2)
    fig.update_yaxes(title_text="Quantiles observés", row=1, col=2)

    st.session_state.graphique_distribution = fig

    # Additional descriptive statistics
    stats_df = pd.DataFrame({
        'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum', 'Skewness', 'Kurtosis'],
        'Valeur': [
            f"{df['y'].mean():.2f}",
            f"{df['y'].median():.2f}",
            f"{df['y'].std():.2f}",
            f"{df['y'].min():.2f}",
            f"{df['y'].max():.2f}",
            f"{df['y'].skew():.2f}",
            f"{df['y'].kurtosis():.2f}"
        ]
    })
    st.session_state.table_stats = stats_df

def process_data(df):
    if df is None:
        return None, None, None

    data = df['y']
    mu = data.mean()
    sigma = data.std()
    _, p_value = stats.normaltest(data)
    
    st.session_state.mu_client = mu
    st.session_state.sigma_client = sigma
    st.session_state.p_value = p_value
    
    return mu, sigma, p_value

def get_excel_inputs():
    uploaded_file = st.file_uploader("Téléchargez votre fichier Excel ou CSV", type=["xlsx", "xls", "csv"])
    if not uploaded_file:
        return

    file_type = uploaded_file.name.split('.')[-1].lower()
    sheet_name = st.text_input("Nom de la feuille Excel (laissez vide pour la première feuille)", "") if file_type == 'xlsx' else None
    sheet_name = None if sheet_name == "" else sheet_name
    date_column = st.text_input("Nom de la colonne des dates", "Date")
    colis_column = st.text_input("Nom de la colonne des ventes/colis", "nombre de colis")
    
    if st.button("Analyser le fichier"):
        df = load_data(uploaded_file, date_column, colis_column, sheet_name)
        if df is not None:
            st.session_state.mu_client, st.session_state.sigma_client, st.session_state.p_value = process_data(df)
            if st.session_state.mu_client is not None:
                st.write(f"Moyenne calculée : {st.session_state.mu_client:.2f}")
                st.write(f"Écart-type calculé : {st.session_state.sigma_client:.2f}")
                st.write(f"P-value du test de normalité : {st.session_state.p_value:.4f}")
                if st.session_state.p_value < 0.05:
                    st.warning("Les données ne suivent probablement pas une distribution normale (p-value < 0.05)")
                else:
                    st.success("Les données suivent probablement une distribution normale (p-value >= 0.05)")
                visualize_data(df)
            else:
                st.error("Erreur lors du traitement des données")

def get_cost_options():
    cost_options = st.checkbox("Ajouter des options de coûts")
    if not cost_options:
        return None, {}

    cost_option = st.selectbox("Sélectionnez une option de coût", ("Coût basé sur les points de collecte", "Coût basé sur le poids"))
    cost_option = "Option 1" if cost_option == "Coût basé sur les points de collecte" else "Option 2"
    
    cost_params = {}
    if cost_option == "Option 1":
        cost_params = {
            "reverse_time": st.slider("Délai avant reverse. Nombre de jours avant que tout les emballages disponibles soient récupérés", min_value=1, max_value=20, value=(1,3), step=1),
            "nb_locations": st.number_input("Nombre de destinations", min_value=0.0, step=1., value=1.0),
            "cost_emballage": st.number_input("Coût d'achat des emballages", min_value=0.0, step=0.01, value=2.0),
            "cost_location": st.number_input("Coût de récupération à un point relai", min_value=0.0, step=0.1, value=5.0),
            "cost_per_demand": st.number_input("Coût par envoi", min_value=0.0, step=0.1, value=1.0)
        }
    elif cost_option == "Option 2":
        cost_params = {
            "reverse_time": st.number_input("Délai avant reverse. Nombre de jours avant que tout les emballages disponibles soient récupérés", value=1.0),
            "cost_emballage": st.number_input("Coût d'achat des emballages", min_value=0.0, step=0.01, value=2.0),
            "poids_sac": st.number_input("Poids moyen", min_value=0.0, step=0.1, value=1.0),
            "cost_per_demand": st.number_input("Coût par envoi par Kg", min_value=0.0, step=0.1, value=1.0),
            "cost_per_return": st.number_input("Coût par retour par Kg", min_value=0.0, step=0.1, value=1.0)
        }

    return cost_option, cost_params

def main():
    st.set_page_config(layout="wide")
    initialiser_session_state()
    st.title("Simulation de Stock")
    menu = st.selectbox("Choix du traitement", ("Traitement par IA a base du fichier Excel", "Traitement par loi normale", "Documentation"))
    
    if menu == "Traitement par loi normale":
        use_file = st.checkbox("Utiliser un fichier Excel ou CSV pour les paramètres de la loi normale")
        
        n_jours = st.number_input("Nombre de jours pour la simulation", min_value=1, max_value=100000, step=1, value=30)
        n_simulations = st.number_input("Nombre de simulations", min_value=1, max_value=100000, step=1, value=400)
        seuil_confiance = st.number_input("Seuil de confiance (en %)", min_value=0.0, max_value=100.0, value=99.0) / 100.0
        
        if use_file:
            get_excel_inputs()
            st.session_state.mu_client = st.number_input("Moyenne du nombre d'expédition par jour (entrepôt)", value=st.session_state.get("mu_client",1000.))
            st.session_state.sigma_client = st.number_input("Ecart-type du nombre d'expédition par jour (entrepôt)", value=st.session_state.get("sigma_client",100.))
        
        else:#if st.session_state.mu_client is None or st.session_state.sigma_client is None:
            st.session_state.mu_client = st.number_input("Moyenne du nombre d'expédition par jour (entrepôt)", value=1000.0)
            st.session_state.sigma_client = st.number_input("Ecart-type du nombre d'expédition par jour (entrepôt)", value=100.0)
        
        
        mu_reverse = st.number_input("Moyenne du délai de restitution (disponibilité de l'emballage)", value=4.0)
        sigma_reverse = st.number_input("Ecart-type du délai de restitution (disponibilité de l'emballage)", value=1.0)
        
        params_client = (st.session_state.mu_client, st.session_state.sigma_client)
        params_reverse = (mu_reverse, sigma_reverse)
        cost_option, cost_params = get_cost_options()

        if st.session_state.graphique_distribution:
            st.plotly_chart(st.session_state.graphique_distribution)
            st.table(st.session_state.table_stats)
        if st.button("Lancer la simulation") and params_client and params_reverse:
            reverse_time = cost_params.get("reverse_time", 1)
            run_simulation(n_simulations, n_jours, params_client, params_reverse, cost_option, cost_params, seuil_confiance, reverse_time)
    elif menu == "Traitement par IA a base du fichier Excel":
        main_excel()
    else:
        show_documentation()

if __name__ == "__main__":
    main()