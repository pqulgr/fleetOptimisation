import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import simulate_, plot_cdf, plot_3_scenarios, plot_return_probs, plot_shipping_demands, specific_fonction_for_accurately_determine_the_cost_of_the_recommended_number_of_bags, plot_cost_vs_reverse, create_summary_table, plot_reverse_optimal_fleet
from documentation import show_documentation
from excel_based_solution import main_excel

def load_data(uploaded_file, date_column, colis_column, sheet_name=None):
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'xlsx':
            if sheet_name is None:
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
        
        return df.rename(columns={date_column: "ds", colis_column: "y"})[["ds","y"]].copy()
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

def visualize_data(df):
    if df is not None:
        # Create a subplot with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogramme et densité des demandes d'export", "Q-Q Plot (Distribution Normale)"))

        # Histogram with KDE
        hist_values, bin_edges = np.histogram(df['y'], bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        kde = stats.gaussian_kde(df['y'])
        
        fig.add_trace(go.Bar(x=bin_centers, y=hist_values, name='Histogram'), row=1, col=1)
        fig.add_trace(go.Scatter(x=bin_centers, y=kde(bin_centers), mode='lines', name='KDE'), row=1, col=1)

        mean = df['y'].mean()
        median = df['y'].median()
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

        st.plotly_chart(fig)

        # Additional descriptive statistics
        st.write("### Statistiques descriptives")
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
        st.table(stats_df)

def process_data(df):
    if df is not None:
        data = df['y']
        mu = data.mean()
        sigma = data.std()
        _, p_value = stats.normaltest(data)
        
        # Store results in session state
        st.session_state.mu_client = mu
        st.session_state.sigma_client = sigma
        st.session_state.p_value = p_value
        
        return mu, sigma, p_value
    return None, None, None


def get_manual_inputs(use_file=False):
    mu_client = sigma_client = None
    if use_file:
        uploaded_file = st.file_uploader("Téléchargez votre fichier Excel ou CSV", type=["xlsx", "xls", "csv"])
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type == 'xlsx':
                sheet_name = st.text_input("Nom de la feuille Excel (laissez vide pour la première feuille)", "")
                sheet_name = None if sheet_name == "" else sheet_name
            else:
                sheet_name = None
            date_column = st.text_input("Nom de la colonne des dates", "Date")
            colis_column = st.text_input("Nom de la colonne des ventes/colis", "nombre de colis")
            
            if st.button("Analyser le fichier"):
                df = load_data(uploaded_file, date_column, colis_column, sheet_name)
                if df is not None:
                    mu_client, sigma_client, p_value = process_data(df)
                    if mu_client is not None:
                        st.write(f"Moyenne calculée : {mu_client:.2f}")
                        st.write(f"Écart-type calculé : {sigma_client:.2f}")
                        st.write(f"P-value du test de normalité : {p_value:.4f}")
                        if p_value < 0.05:
                            st.warning("Les données ne suivent probablement pas une distribution normale (p-value < 0.05)")
                        else:
                            st.success("Les données suivent probablement une distribution normale (p-value >= 0.05)")
                        
                        visualize_data(df)
                    else:
                        st.error("Erreur lors du traitement des données")
    
    if mu_client is None or sigma_client is None:
        mu_client = st.number_input("Moyenne du nombre d'expédition par jour (entrepôt)", value=st.session_state.get('mu_client', 1000.0))
        sigma_client = st.number_input("Ecart-type du nombre d'expédition par jour (entrepôt)", value=st.session_state.get('sigma_client', 100.0))
    
    mu_reverse = st.number_input("Moyenne du délai de restitution (disponibilité de l'emballage)", value=4.0)
    sigma_reverse = st.number_input("Ecart-type du délai de restitution (disponibilité de l'emballage)", value=1.0)
    return (mu_client, sigma_client), (mu_reverse, sigma_reverse)

def get_cost_options():
    cost_options = st.checkbox("Ajouter des options de coûts")
    cost_params = {}
    cost_option = None
    if cost_options:
        cost_option = st.selectbox("Sélectionnez une option de coût", ("Coût basé sur les points de collecte", "Coût basé sur le poids"))
        cost_option = "Option 1" if cost_option=="Coût basé sur les points de collecte" else "Option 2"
        if cost_option == "Option 1":
            cost_params["reverse_time"] = st.slider("Délai avant reverse. Nombre de jours avant que tout les emballages disponibles soient récupérés", min_value=1, max_value=20, value=(1,3), step=1, label_visibility="visible")
            cost_params["nb_locations"] = st.number_input("Nombre de destinations", min_value=0.0, step=1., value=1.0)
            cost_params["cost_emballage"] = st.number_input("Coût d'achat des emballages", min_value=0.0, step=0.01, value=2.0)
            cost_params["cost_location"] = st.number_input("Coût de récupération à un point relai", min_value=0.0, step=0.1, value=5.0)
            cost_params["cost_per_demand"] = st.number_input("Coût par envoi", min_value=0.0, step=0.1, value=1.0)
        elif cost_option == "Option 2":
            cost_params["reverse_time"] = st.number_input("Délai avant reverse. Nombre de jours avant que tout les emballages disponibles soient récupérés", value=1.0)
            cost_params["cost_emballage"] = st.number_input("Coût d'achat des emballages", min_value=0.0, step=0.01, value=2.0)
            cost_params["poids_sac"] = st.number_input("Poids moyen", min_value=0.0, step=0.1, value=1.0)
            cost_params["cost_per_demand"] = st.number_input("Coût par envoi par Kg", min_value=0.0, step=0.1, value=1.0)
            cost_params["cost_per_return"] = st.number_input("Coût par retour par Kg", min_value=0.0, step=0.1, value=1.0)

    return cost_option, cost_params

def run_simulation(n_simulations, n_jours, params_client, params_reverse, cost_option, cost_params, seuil_confiance, reverse_time):
    if cost_option=="Option 1":
        y = []
        seuil_x_values = []
        reverse_range = range(reverse_time[0], reverse_time[1]+1)
        for i in reverse_range:
            results, resultats_3_simulations, demand_scenarios, returns_scenarios, return_probs, shipping_demands, costs = simulate_(
                n_simulations, n_jours, params_client, params_reverse, i, cost_option, cost_params)
            
            st.markdown(f"## Reverse définie à {i} jours")
            seuil_x = plot_cdf(results, seuil_confiance)
            seuil_x_values.append(seuil_x)
            y.append((seuil_x,costs.get(str(seuil_x))))
        
        costs_per_reverse = specific_fonction_for_accurately_determine_the_cost_of_the_recommended_number_of_bags(
            y, cost_params, 400, params_reverse, params_client, n_jours, cost_option
        )
        plot_cost_vs_reverse(costs_per_reverse)
        plot_reverse_optimal_fleet(reverse_range, seuil_x_values)
        create_summary_table(reverse_range, costs_per_reverse, seuil_x_values)

    elif cost_option=="Option 2":
        results, resultats_3_simulations, demand_scenarios, returns_scenarios, return_probs, shipping_demands, costs = simulate_(
            n_simulations, n_jours, params_client, params_reverse, reverse_time, cost_option, cost_params)
        
        seuil_x = plot_cdf(results, seuil_confiance)
        plot_3_scenarios(resultats_3_simulations, demand_scenarios, returns_scenarios)

        rever = int(reverse_time)
        y = np.array([(seuil_x,costs.get(str(seuil_x)))])
        costs_per_reverse = specific_fonction_for_accurately_determine_the_cost_of_the_recommended_number_of_bags(
            y, cost_params, 400, params_reverse, params_client, n_jours, cost_option
        )

        plot_cost_vs_reverse(costs_per_reverse)
        create_summary_table([rever], costs_per_reverse, [seuil_x])

    else:
        results, resultats_3_simulations, demand_scenarios, returns_scenarios, return_probs, shipping_demands, costs = simulate_(
            n_simulations, n_jours, params_client, params_reverse, reverse_time, cost_option, cost_params)
        
        plot_cdf(results, seuil_confiance)
        plot_3_scenarios(resultats_3_simulations, demand_scenarios, returns_scenarios)
        plot_return_probs(return_probs, params_reverse)
        plot_shipping_demands(shipping_demands, params_client)

def main():
    st.set_page_config(layout="wide")
    
    st.title("Simulation de Stock")
    menu = st.selectbox("Choix du traitement", ("Traitement par IA a base du fichier Excel", "Traitement par loi normale", "Documentation"))
    
    if menu == "Traitement par loi normale":
        use_file = st.checkbox("Utiliser un fichier Excel ou CSV pour les paramètres de la loi normale")
        
        n_jours = st.number_input("Nombre de jours pour la simulation", min_value=1, step=1, value=30)
        n_simulations = st.number_input("Nombre de simulations", min_value=1, step=1, value=400)
        seuil_confiance = st.number_input("Seuil de confiance (en %)", min_value=0.0, max_value=100.0, value=95.0) / 100.0
            
        params_client, params_reverse = get_manual_inputs(use_file)
        cost_option, cost_params = get_cost_options()
            
        if st.button("Lancer la simulation") and params_client and params_reverse:
            if cost_option == "Option 1":
                reverse_time = cost_params["reverse_time"]
            else:
                reverse_time = cost_params.get("reverse_time", 1)
            run_simulation(n_simulations, n_jours, params_client, params_reverse, cost_option, cost_params, seuil_confiance, reverse_time)
    elif menu == "Traitement par IA a base du fichier Excel":
        main_excel()
    else:
        show_documentation()

if __name__ == "__main__":
    main()