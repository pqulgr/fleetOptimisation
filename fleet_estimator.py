import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

class FleetEstimator:
    def __init__(self, true_data, predictions, cost_option, cost_params):
        self.true_data = true_data
        self.predictions = predictions
        self.combined_data = self._combine_data()
        self.cost_option = cost_option
        self.cost_params = cost_params
        self.fleet_size = None
        self.best_reverse_time = None
        self.results = {}

    def _combine_data(self):
        combined = pd.concat([self.true_data, self.predictions])
        combined = combined.sort_values('ds').reset_index(drop=True)
        return combined

    def estimate_fleet_size(self):
        if self.combined_data.empty:
            raise ValueError("Aucune donnée n'est disponible pour l'estimation.")

        if 'y' not in self.combined_data.columns and 'yhat1' not in self.combined_data.columns:
            raise ValueError("Ni la colonne 'y' ni 'yhat1' n'est présente dans les données.")

        self.combined_data['demand'] = self.combined_data['yhat1'].fillna(self.combined_data['y'])

        n_days = len(self.combined_data)
        demand = self.combined_data['demand'].values

        if self.cost_option == "Option 1":
            reverse_range = range(self.cost_params["reverse_time"][0], self.cost_params["reverse_time"][1] + 1)
            best_cost = float('inf')
            for reverse_time in reverse_range:
                stock, pending_returns = self._calculate_stock(n_days, demand, reverse_time)
                fleet_size = int(np.ceil(np.max(np.abs(stock))))
                cost = self._calculate_total_cost(fleet_size, pending_returns, demand, reverse_time)
                self.results[reverse_time] = {'fleet_size': fleet_size, 'cost': cost, 'stock': stock, 'pending_returns': pending_returns}
                if cost < best_cost:
                    best_cost = cost
                    self.best_reverse_time = reverse_time
                    self.fleet_size = fleet_size

            self.combined_data['stock'] = self.results[self.best_reverse_time]['stock']
            self.combined_data['pending_returns'] = self.results[self.best_reverse_time]['pending_returns']

        elif self.cost_option == "Option 2":
            reverse_time = self.cost_params["reverse_time"]
            stock, pending_returns = self._calculate_stock(n_days, demand, reverse_time)
            self.fleet_size = int(np.ceil(np.max(np.abs(stock))))
            cost = self._calculate_total_cost(self.fleet_size, pending_returns, demand, reverse_time)
            self.results[reverse_time] = {'fleet_size': self.fleet_size, 'cost': cost, 'stock': stock, 'pending_returns': pending_returns}
            self.best_reverse_time = reverse_time

            self.combined_data['stock'] = stock
            self.combined_data['pending_returns'] = pending_returns

        return self.fleet_size

    def _calculate_stock(self, n_days, demand, reverse_time):
        stock = np.zeros(n_days)
        pending_returns = np.zeros(n_days)
        available_returns = np.zeros(n_days)

        for day in range(n_days):
            if day > 0:
                future_days = reverse_time
                np.add.at(available_returns, future_days, demand[day - 1])

            if day % reverse_time == 0:
                pending_returns[day] += np.sum(available_returns[:day+1])
                available_returns[:day+1] = 0

            if day > 0:
                stock[day] = stock[day-1] + pending_returns[day] - demand[day]

        return stock, pending_returns

    def _calculate_total_cost(self, fleet_size, pending_returns, demand, reverse_time):
        if self.cost_option == "Option 1":
            cost_dem = self.cost_params["cost_per_demand"]
            nb_locations = self.cost_params["nb_locations"]
            cost_location = self.cost_params["cost_location"]
            cost_emballage = self.cost_params["cost_emballage"]
            
            cost = len(pending_returns[pending_returns > 0]) * cost_location * nb_locations + fleet_size * cost_emballage
            cost += sum(demand * cost_dem)

        elif self.cost_option == "Option 2":
            cost_dem_kg = self.cost_params["cost_per_demand"]
            cost_ret_kg = self.cost_params["cost_per_return"]
            poids_sac = self.cost_params["poids_sac"]
            cost_emballage = self.cost_params["cost_emballage"]
            
            cost = cost_emballage * fleet_size
            cost += sum(cost_dem_kg * poids_sac * demand + cost_ret_kg * poids_sac * pending_returns)

        return cost

    def plot_stock_over_time(self, reverse_time=None):
        if reverse_time is None:
            reverse_time = self.best_reverse_time

        data = self.results[reverse_time]
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.combined_data['ds'], y=data['stock'], mode='lines', name='Stock'))
        fig.add_trace(go.Scatter(x=self.combined_data['ds'], y=self.combined_data['demand'], mode='lines', name='Demande'))
        fig.add_trace(go.Scatter(x=self.combined_data['ds'], y=data['pending_returns'], mode='lines', name='Retours'))

        fig.update_layout(
            title=f'Évolution du stock, de la demande et des retours au fil du temps (Reverse: {reverse_time} jours)',
            xaxis_title='Date',
            yaxis_title='Quantité',
            hovermode='x'
        )

        return fig

    def create_summary_table(self):
        data = []
        for reverse_time, result in self.results.items():
            data.append({
                'Délai de retour': reverse_time,
                'Nombre d\'emballages': result['fleet_size'],
                'Coût estimé': f"{result['cost']:.0f}"
            })

        df = pd.DataFrame(data)
        return df

    def display_results(self):
        st.plotly_chart(self.plot_stock_over_time())
        
        st.markdown("## Tableau récapitulatif:")
        st.dataframe(self.create_summary_table())

        st.markdown("## Résultat de l'estimation:")
        st.markdown(f"**Meilleur délai de retour:** {self.best_reverse_time} jours")
        st.markdown(f"**Nombre d'emballages estimé:** {self.fleet_size}")
        st.markdown(f"**Coût estimé:** {self.results[self.best_reverse_time]['cost']:.0f}€")