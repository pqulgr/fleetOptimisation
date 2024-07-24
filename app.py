import streamlit as st
import numpy as np
from utils import simulate_, plot_cdf, plot_3_scenarios, plot_return_probs, plot_shipping_demands, specific_fonction_for_accurately_determine_the_cost_of_the_recommended_number_of_bags, plot_cost_vs_reverse, create_summary_table, plot_reverse_optimal_fleet
from documentation import show_documentation
from excel_based_solution import main_excel

def get_manual_inputs():
    mu_client = st.number_input("Moyenne du nombre d'expédition par jour (entrepôt)", value=1000.0)
    sigma_client = st.number_input("Ecart-type du nombre d'expédition par jour (entrepôt)", value=100.0)
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
    
    menu = ["Simulation", "Documentation"]
    st.set_page_config(layout="wide")
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice=="Simulation":
        st.title("Simulation de Stock")
        excel_based = st.selectbox("Choix du traitement", ("Traitement par IA a base du fichier Excel", "Traitement par loi normale"))
        if excel_based=="Traitement par IA a base du fichier Excel":
            main_excel()
        else:
            n_jours = st.number_input("Nombre de jours pour la simulation", min_value=1, step=1, value=30)
            n_simulations = st.number_input("Nombre de simulations", min_value=1, step=1, value=400)
            seuil_confiance = st.number_input("Seuil de confiance (en %)", min_value=0.0, max_value=100.0, value=95.0) / 100.0
            
            params_client, params_reverse = get_manual_inputs()
            cost_option, cost_params = get_cost_options()
            
            if st.button("Lancer la simulation") and params_client and params_reverse:
                if cost_params.get("reverse_time"):
                    reverse_time = cost_params["reverse_time"]
                else:
                    reverse_time = 1
                run_simulation(n_simulations, n_jours, params_client, params_reverse, cost_option, cost_params, seuil_confiance, reverse_time)
    if choice == "Documentation":
        show_documentation()

if __name__ == "__main__":
    main()