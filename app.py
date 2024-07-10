import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import simulate_, plot_cdf, plot_3_scenarios, plot_return_probs, plot_shipping_demands, specific_fonction_for_accurately_determine_the_cost_of_the_recommended_number_of_bags

def load_excel_data():
    from excel_based_solution import load_data
    st.title("Analyse de données d'expédition")
    uploaded_file = st.file_uploader("Choisissez un fichier Excel ou CSV", type=["xlsx", "csv"])
    if uploaded_file is not None:
        mu_client, sigma_client, mu_reverse, sigma_reverse = load_data(uploaded_file)
        return (mu_client, sigma_client), (mu_reverse, sigma_reverse)
    return None, None

def get_manual_inputs():
    mu_client = st.number_input("Moyenne de la quantité d'exports (entrepôt)", value=2.0)
    sigma_client = st.number_input("Ecart-type demande (entrepôt)", value=1.0)
    mu_reverse = st.number_input("Moyenne du nombre de jours écoulés avant disponibilité (reverse)", value=10.0)
    sigma_reverse = st.number_input("Ecart-type du nombre de jours écoulés avant disponibilité (reverse)", value=1.0)
    return (mu_client, sigma_client), (mu_reverse, sigma_reverse)

def get_cost_options():
    cost_options = st.checkbox("Ajouter des options de coûts")
    cost_params = {}
    cost_option = None
    if cost_options:
        cost_option = st.selectbox("Sélectionnez une option de coût", ("Option 1", "Option 2"))
        if cost_option == "Option 1":
            cost_params["reverse_time"] = st.slider("Délai du retour des camions (reverse)", min_value=1, max_value=20, value=(1,1), step=1, label_visibility="visible")
            cost_params["nb_locations"] = st.number_input("Nombre de destinations", min_value=0.0, step=1., value=1.0)
            cost_params["cost_emballage"] = st.number_input("Coût d'achat des emballages", min_value=0.0, step=0.01, value=2.0)
            cost_params["cost_location"] = st.number_input("Coût d'initialisation d'un point relai", min_value=0.0, step=0.1, value=5.0)
            cost_params["cost_per_demand"] = st.number_input("Coût par envoi", min_value=0.0, step=0.1, value=1.0)
            #cost_params["cost_per_return"] = st.number_input("Coût par retour", min_value=0.0, step=0.1, value=0.5)
        elif cost_option == "Option 2":
            cost_params["reverse_time"] = st.number_input("Délai du retour des camions (reverse)", value=1.0)
            cost_params["cost_emballage"] = st.number_input("Coût d'achat des emballages", min_value=0.0, step=0.01, value=2.0)
            cost_params["poids_sac"] = st.number_input("Poids moyen", min_value=0.0, step=0.1, value=1.0)
            cost_params["cost_per_demand"] = st.number_input("Coût par envoi par Kg", min_value=0.0, step=0.1, value=1.0)
            cost_params["cost_per_return"] = st.number_input("Coût par retour par Kg", min_value=0.0, step=0.1, value=1.0)

    return cost_option, cost_params

def run_simulation(n_simulations, n_jours, params_client, params_reverse, cost_option, cost_params, seuil_confiance, reverse_time):
    if cost_option=="Option 1":
        y=[]
        for i in range(reverse_time[0],reverse_time[1]+1):
            results, resultats_3_simulations, demand_scenarios, returns_scenarios, return_probs, shipping_demands, costs = simulate_(
                n_simulations, n_jours, params_client, params_reverse, i, cost_option, cost_params)
            
            st.write(f"reverse définie à {i} jours")
            seuil_x = plot_cdf(results, seuil_confiance)
            #plot_3_scenarios(resultats_3_simulations, demand_scenarios, returns_scenarios)
            y.append(costs.get(str(seuil_x)))
        
        y = np.array(specific_fonction_for_accurately_determine_the_cost_of_the_recommended_number_of_bags(
            y, cost_params, 50, params_reverse, params_client, n_jours
        ))


        fig, ax = plt.subplots()
        ax.plot(range(reverse_time[0],reverse_time[1]+1), y)
        ax.set_xlabel('reverse')
        ax.set_ylabel('coût')
        ax.set_title('Distribution des coûts totaux')
        plt.tight_layout()
        st.pyplot(fig)
    elif cost_option=="Option 2":
        results, resultats_3_simulations, demand_scenarios, returns_scenarios, return_probs, shipping_demands, costs = simulate_(
            n_simulations, n_jours, params_client, params_reverse, reverse_time, cost_option, cost_params)
        
        seuil_x = plot_cdf(results, seuil_confiance)
        plot_3_scenarios(resultats_3_simulations, demand_scenarios, returns_scenarios)
        #plot_return_probs(return_probs, params_reverse)
        #plot_shipping_demands(shipping_demands, params_client)

        st.write("Coûts totaux des simulations:")
        st.write(costs)
        fig, ax = plt.subplots()
        ax.hist(costs, bins=20)
        ax.set_xlabel('Coût total')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution des coûts totaux')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        results, resultats_3_simulations, demand_scenarios, returns_scenarios, return_probs, shipping_demands, costs = simulate_(
            n_simulations, n_jours, params_client, params_reverse, reverse_time, cost_option, cost_params)
        
        plot_cdf(results, seuil_confiance)
        plot_3_scenarios(resultats_3_simulations, demand_scenarios, returns_scenarios)
        plot_return_probs(return_probs, params_reverse)
        plot_shipping_demands(shipping_demands, params_client)

def main():
    st.title("Simulation de Stock")
    
    excel_based = st.checkbox("Utiliser un fichier Excel")
    n_jours = st.number_input("Nombre de jours pour la simulation", min_value=1, step=1, value=30)
    n_simulations = st.number_input("Nombre de simulations", min_value=1, step=1, value=400)
    seuil_confiance = st.number_input("Seuil de confiance (en %)", min_value=0.0, max_value=100.0, value=95.0) / 100.0
    
    if excel_based:
        cost_params["reverse_time"] = st.number_input("Délai du retour des camions (reverse)", value=1.0)
        params_client, params_reverse = load_excel_data()
    else:
        params_client, params_reverse = get_manual_inputs()
    
    cost_option, cost_params = get_cost_options()
    
    if st.button("Lancer la simulation") and params_client and params_reverse:
        if cost_params.get("reverse_time"):
            reverse_time = cost_params["reverse_time"]
        else:
            reverse_time = 1
        run_simulation(n_simulations, n_jours, params_client, params_reverse, cost_option, cost_params, seuil_confiance, reverse_time)

if __name__ == "__main__":
    main()