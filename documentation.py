import streamlit as st

def show_documentation():
    st.markdown("""
    # Documentation de l'application de simulation de stock d'emballages réutilisables

    ## Introduction
    Cette application simule la gestion d'un stock d'emballages réutilisables dans un système de chaîne d'approvisionnement en boucle fermée. Elle aide à déterminer le nombre optimal d'emballages nécessaires pour un entrepôt qui expédie des commandes à plusieurs magasins.

    ## Contexte du problème
    Un entrepôt reçoit des demandes quotidiennes de commandes à expédier vers différents magasins. Une fois les commandes livrées et les produits déballés, les emballages vides sont collectés périodiquement par un camion et renvoyés à l'entrepôt. L'objectif est de déterminer la quantité optimale d'emballages réutilisables à maintenir en stock pour répondre à la demande tout en minimisant les coûts.

    ## Paramètres principaux

    ### Nombre de jours et nombre de simulations
    - **Description** : Durée totale de la période simulée, en jours. Précision des résultats
    - **Pourquoi c'est important** : Une période plus longue impactent le coût final et la précision des calculs, mais augmente le temps de calcul.

    ### Seuil de confiance
    - **Description** : Niveau de confiance pour déterminer le stock optimal.
    - **Pourquoi c'est important** : Un seuil plus élevé réduit le risque de pénurie mais augmente la flotte nécessaire'.

    ## Paramètres de distribution

    ### Paramètres client (expédition)
    - **Moyenne du nombre d'expédition par jour** : Nombre moyen d'emballages expédiés quotidiennement.
    - **Écart-type du nombre d'expédition par jour** : Mesure de la variabilité des expéditions quotidiennes.
    - **Pourquoi c'est important** : Ces paramètres définissent le profil de la demande. Une demande plus variable nécessite généralement plus de stock.

    ### Paramètres de retour
    - **Moyenne du délai de disponibilité** : Temps moyen (en jours) avant qu'un emballage soit disponible pour être récupéré.
    - **Écart-type du délai de disponibilité** : Mesure de la variabilité du temps de disponibilité.
    - **Pourquoi c'est important** : Des délais de disponibilité plus longs ou plus variables nécessitent généralement plus de stock.

    ## Options de coût

    ### Option 1 (Coût basé sur les points de collecte)
    - **Délai avant reverse** : Fréquence de collecte des emballages (en jours). 1 signifie que le camion ramène tout les jours les emballages disponibles dans tout les magasins
    - **Nombre de destinations** : Nombre de points de livraison différents. Avec cette option, lors d'une ramasse, tout les lieux vont facturer leurs coût de service afn que le camion récupère l'emballage.
    - **Coût de récupération à un point relai** : Coût associé à la collecte d'un emballage.
    - **Coût d'achat des emballages** : Prix unitaire d'un emballage.
    - **Coût par envoi** : Coût fixe pour chaque expédition par emballage.

    ### Option 2 (Coût basé sur le poids)
    - **Délai avant reverse** : Identique à l'Option 1.
    - **Coût d'achat des emballages** : Identique à l'Option 1.
    - **Poids moyen** : Poids moyen d'un emballage en kg.
    - **Coût par envoi par Kg** : Coût d'expédition par kilogramme.
    - **Coût par retour par Kg** : Coût de retour par kilogramme.

    ## Utilisation de fichiers Excel/CSV
    Vous pouvez importer des données historiques à partir de fichiers Excel ou CSV. Ces fichiers doivent contenir au minimum deux colonnes :
    - "Export" : Nombre d'emballages expédiés chaque jour
    - "Retour" : Nombre d'emballages retournés chaque jour

    ## Résultats de la simulation
    - **Fonction de Répartition (CDF)** : Montre la probabilité de différents niveaux de stock.
    - **Scénarios de simulation** : Affiche les scénarios de demande faible, moyenne et forte.
    - **Distribution des coûts** : Visualise comment les coûts varient selon le délai de retour.
    - **Évolution du stock optimal** : Montre comment le stock optimal change selon le délai de retour.
    - **Tableau récapitulatif** : Résume les coûts moyens pour différentes combinaisons de paramètres.

    ## Interprétation des résultats
    Les résultats vous aideront à déterminer :
    1. Le nombre optimal d'emballages à maintenir en stock.
    2. Option 1 : le nombre de reverse à parametrer pour minimiser le coût.

    Utilisez ces informations pour prendre des décisions éclairées sur la gestion de votre stock d'emballages réutilisables.
    """)