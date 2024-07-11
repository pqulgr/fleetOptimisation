import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

def f_option_1(params, returns, demand):
    cost_dem = params["cost_per_demand"]
    nb_locations = params["nb_locations"]
    nb_emballage = params["nb_emballages"]
    cost_location = params["cost_location"]
    cost_emballage = params["cost_emballage"]
    cost = len(returns[returns > 0]) * cost_location * nb_locations + nb_emballage * cost_emballage
    cost += sum(demand * cost_dem)
    return cost

def f_option_2(params, returns, demand):
    cost=0
    cost_dem_kg = params["cost_per_demand"]
    cost_ret_kg = params["cost_per_return"]
    poids_sac = params["poids_sac"]
    nb_emballage = params["nb_emballages"]
    cost_emballage = params["cost_emballage"]
    cost += cost_emballage * nb_emballage
    cost += sum(cost_dem_kg * poids_sac * demand + cost_ret_kg * poids_sac * returns)
    return cost



def simulate_(n_sim, n_days, params_client, params_reverse, reverse_time, cost_option=None, cost_params=None):
    length_return = int(max(n_days, params_reverse[0]+1+5*params_reverse[1]))
    prob = np.array([norm.cdf(k + 0.5, params_reverse[0], params_reverse[1]) - norm.cdf(k - 0.5, params_reverse[0], params_reverse[1]) for k in range(1, length_return)])
    return_probs = prob / sum(prob)

    length_shipping = int(max(n_days, params_client[0]+1+5*params_client[1]))
    prob = np.array([norm.cdf(k + 0.5, params_client[0], params_client[1]) - norm.cdf(k - 0.5, params_client[0], params_client[1]) for k in range(length_shipping)])
    shipping_demands = prob / sum(prob)

    store_results = {"minimum": np.array([]), "medium": np.array([]), "maximum": np.array([])}
    store_min = np.inf
    store_max = -np.inf
    
    final_stock_levels = []
    demand_scenarios = {"minimum": None, "medium": None, "maximum": None}
    returns_scenarios = {"minimum": None, "medium": None, "maximum": None}
    costs = {}

    progress_bar = st.empty()
        
    for i in range(n_sim):
        progress_bar.progress((i + 1) / n_sim)

        stock_warehouse = np.zeros(n_days)
        pending_returns = np.zeros(n_days)
        available_returns = np.zeros(n_days)
        demand = np.random.choice(np.arange(length_shipping), size=n_days, p=shipping_demands)
        
        for day in range(n_days):
            day_reverse_choices = np.random.choice(np.arange(1, length_return), size=demand[day], p=return_probs)
            future_days = day + day_reverse_choices
            future_days = future_days[future_days < n_days]
            np.add.at(available_returns, future_days, 1)  # Add to available returns instead of pending returns
            
            # Check if it's a day for the truck to collect returns
            if day % reverse_time == 0:
                # Add all available returns to pending returns
                pending_returns[day] += np.sum(available_returns[:day+1])
                available_returns[:day+1] = 0  # Reset available returns
        
        returns = pending_returns.copy()
        stock_warehouse = np.cumsum(pending_returns - demand)

        stock_final = (-stock_warehouse).max()
        if store_results["medium"].size == 0 or (store_min + store_max) // 2 == stock_final:
            store_results["medium"] = -stock_warehouse
            demand_scenarios["medium"] = demand
            returns_scenarios["medium"] = returns
        if stock_final <= store_min:
            store_min = stock_final
            store_results["minimum"] = -stock_warehouse
            demand_scenarios["minimum"] = demand
            returns_scenarios["minimum"] = returns
        if stock_final >= store_max:
            store_max = stock_final
            store_results["maximum"] = -stock_warehouse
            demand_scenarios["maximum"] = demand
            returns_scenarios["maximum"] = returns
        final_stock_levels.append(stock_final)
        if cost_option=="Option 1":
            cost_params["nb_emballages"] = stock_final
            costs[str(stock_final)] = f_option_1(cost_params, returns, demand)
        elif cost_option=="Option 2":
            cost_params["nb_emballages"] = stock_final
            costs[str(stock_final)] = f_option_2(cost_params, returns, demand)
    progress_bar.empty()
    return final_stock_levels, store_results, demand_scenarios, returns_scenarios, return_probs, shipping_demands, costs

def plot_cdf(results, seuil_confiance=0.95):
    sorted_results = np.sort(results)
    yvals = np.arange(1, len(sorted_results) + 1) / len(sorted_results)
    
    seuil_ = np.where(yvals >= seuil_confiance)[0][0]
    seuil_x = sorted_results[seuil_]
    seuil_y = yvals[seuil_]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sorted_results, y=yvals, mode='lines', name='F(X<=k)'))
    fig.add_trace(go.Scatter(x=[seuil_x], y=[seuil_y], mode='markers', name='Seuil', marker=dict(size=10)))
    
    fig.add_shape(type="line", x0=min(sorted_results), x1=max(sorted_results), 
                  y0=seuil_confiance, y1=seuil_confiance, line=dict(color="green", dash="dash"))
    
    fig.add_shape(type="line", x0=seuil_x, x1=seuil_x, 
                  y0=0, y1=seuil_y, line=dict(color="blue", dash="dash"))
    
    fig.update_layout(
        title='Fonction de Répartition (CDF)',
        xaxis_title='Quantité de stock restante (k)',
        yaxis_title='Probabilité cumulée',
        annotations=[
            dict(x=seuil_x, y=seuil_y, xref="x", yref="y",
                 text=f'({seuil_x:.0f}, {seuil_y:.2f})', showarrow=True, arrowhead=7, ax=0, ay=-40)
        ]
    )
    
    st.plotly_chart(fig)
    return seuil_x

def plot_3_scenarios(dic_scenes, demand_scenarios, returns_scenarios):
    scenarios = ['minimum', 'medium', 'maximum']
    titles = ['Scénario à faible demande', 'Scénario à moyenne demande', 'Scénario à forte demande']
    
    fig = go.Figure()
    
    for i, scenario in enumerate(scenarios):
        y = dic_scenes[scenario]
        X = np.arange(1, len(y) + 1)
        
        fig.add_trace(go.Scatter(x=X, y=y, mode='lines', name=f'Stock ({scenario})', visible=i==1))
        fig.add_trace(go.Scatter(x=X, y=demand_scenarios[scenario], mode='lines', name=f'Demande ({scenario})', visible=i==1))
        fig.add_trace(go.Scatter(x=X, y=returns_scenarios[scenario], mode='lines', name=f'Retours ({scenario})', visible=i==1))
    
    fig.update_layout(
        updatemenus=[
            dict(
                active=1,
                buttons=list([
                    dict(label="Faible demande",
                         method="update",
                         args=[{"visible": [True, True, True, False, False, False, False, False, False]},
                               {"title": "Scénario à faible demande"}]),
                    dict(label="Moyenne demande",
                         method="update",
                         args=[{"visible": [False, False, False, True, True, True, False, False, False]},
                               {"title": "Scénario à moyenne demande"}]),
                    dict(label="Forte demande",
                         method="update",
                         args=[{"visible": [False, False, False, False, False, False, True, True, True]},
                               {"title": "Scénario à forte demande"}]),
                ]),
            )
        ]
    )
    
    fig.update_layout(
        xaxis_title='Jours',
        yaxis_title='Quantité',
        title='Scénarios de simulation'
    )
    
    st.plotly_chart(fig)

def plot_return_probs(return_probs, params):
    n_min = max(1, int(params[0] - 5 * params[1]))
    n_max = min(int(params[0] + 5 * params[1]), len(return_probs))
    
    if n_min < n_max:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_min, n_max),
            y=return_probs[n_min-1:n_max-1],
            mode='lines',
            name='Probabilité de retour'
        ))
        
        fig.update_layout(
            title='Probabilité de retour par jour',
            xaxis_title='Jours',
            yaxis_title='Probabilité',
            hovermode='x'
        )
        
        st.plotly_chart(fig)
    else:
        st.write("Les paramètres de centrage de la courbe de retour sont incorrects. Veuillez vérifier les paramètres de la distribution.")

def plot_shipping_demands(shipping_demands, params):
    n_min = max(0, int(params[0] - 5 * params[1]))
    n_max = min(int(params[0] + 5 * params[1]), len(shipping_demands))
    
    if n_min < n_max:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_min, n_max),
            y=shipping_demands[n_min:n_max],
            mode='lines',
            name='Probabilité de demande'
        ))
        
        fig.update_layout(
            title='Probabilité de demande par jour',
            xaxis_title='Jours',
            yaxis_title='Probabilité',
            hovermode='x'
        )
        
        st.plotly_chart(fig)
    else:
        st.write("Les paramètres de centrage de la courbe de demande sont incorrects. Veuillez vérifier les paramètres de la distribution.")

def plot_cost_vs_reverse(reverse_range, costs):
    reverse_range = list(reverse_range)  # Convertir range en liste

    # Ajout d'un boxplot pour montrer la distribution des coûts
    fig = go.Figure()
    for i, reverse in enumerate(reverse_range):
        fig.add_trace(go.Box(y=costs[i], name=str(reverse)))
    if len(reverse_range)==1:
        fig.update_layout(
            title='Distribution du coût en fonction du délai de retour',
            xaxis_title='Délai de retour (jours)',
            yaxis_title='Coût'
        )
        st.plotly_chart(fig)
    else:
        fig.update_layout(
            title='Distribution des coûts en fonction du délai de retour',
            xaxis_title='Délai de retour (jours)',
            yaxis_title='Coût'
        )
        st.plotly_chart(fig)

def specific_fonction_for_accurately_determine_the_cost_of_the_recommended_number_of_bags(y, cost_params, n_sim, params_reverse, params_client, n_days_sim, cost_option):
    y_ = [[] for _ in range(len(y))]

    length_return = int(max(n_days_sim, params_reverse[0]+1+5*params_reverse[1]))
    prob = np.array([norm.cdf(k + 0.5, params_reverse[0], params_reverse[1]) - norm.cdf(k - 0.5, params_reverse[0], params_reverse[1]) for k in range(1, length_return)])
    return_probs = prob / sum(prob)

    length_shipping = int(max(n_days_sim, params_client[0]+1+5*params_client[1]))
    prob = np.array([norm.cdf(k + 0.5, params_client[0], params_client[1]) - norm.cdf(k - 0.5, params_client[0], params_client[1]) for k in range(length_shipping)])
    shipping_demands = prob / sum(prob)
    progress_bar = st.empty()
    for i,item in enumerate(y):
        progress_bar.progress((i + 1) / len(y))
        cost_params["nb_emballages"] = item
            
        for j in range(n_sim):

            pending_returns = np.zeros(n_days_sim)
            available_returns = np.zeros(n_days_sim)  # New array to track available returns
            demand = np.random.choice(np.arange(length_shipping), size=n_days_sim, p=shipping_demands)
            
            for day in range(n_days_sim):
                day_reverse_choices = np.random.choice(np.arange(1, length_return), size=demand[day], p=return_probs)
                future_days = day + day_reverse_choices
                future_days = future_days[future_days < n_days_sim]
                np.add.at(available_returns, future_days, 1)  # Add to available returns instead of pending returns
                
                # Check if it's a day for the truck to collect returns
                if day % (j+1) == 0:
                    # Add all available returns to pending returns
                    pending_returns[day] += np.sum(available_returns[:day+1])
                    available_returns[:day+1] = 0  # Reset available returns
            
            returns = pending_returns.copy()
            if cost_option == "Option 1":
                y_[i].append(f_option_1(cost_params, returns, demand))
            elif cost_option == "Option 2":
                y_[i].append(f_option_2(cost_params, returns, demand))
        
    progress_bar.empty()
    return y_