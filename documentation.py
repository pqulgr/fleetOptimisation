import streamlit as st

def show_documentation():
    st.markdown("""
# Documentation : Méthodes de sizing de flotte d'emballages réutilisables

## Sommaire

1. [Introduction](#1-introduction)
2. [Aperçu des méthodes de sizing](#2-aperçu-des-méthodes-de-sizing)
3. [Méthode basée sur la simulation (sans import Excel)](#3-méthode-basée-sur-la-simulation-sans-import-excel)
   3.1 [Principes de la méthode](#31-principes-de-la-méthode)
   3.2 [Paramètres d'entrée](#32-paramètres-dentrée)
   3.3 [Processus de simulation](#33-processus-de-simulation)
   3.4 [Interprétation des résultats](#34-interprétation-des-résultats)
   3.5 [Avantages et limites](#35-avantages-et-limites)
4. [Méthode basée sur l'analyse des données historiques (avec import Excel)](#4-méthode-basée-sur-lanalyse-des-données-historiques-avec-import-excel)
   4.1 [Principes de la méthode](#41-principes-de-la-méthode)
   4.2 [Import et prétraitement des données](#42-import-et-prétraitement-des-données)
   4.3 [Analyse et modélisation](#43-analyse-et-modélisation)
   4.4 [Estimation de la flotte](#44-estimation-de-la-flotte)
   4.5 [Avantages et limites](#45-avantages-et-limites)
5. [Choix de la méthode appropriée](#5-choix-de-la-méthode-appropriée)
6. [Optimisation des coûts](#6-optimisation-des-coûts)
7. [Conclusion](#7-conclusion)

## 1. Introduction

Le sizing de flotte d'emballages réutilisables est un défi crucial pour les entreprises logistiques cherchant à optimiser leurs opérations et réduire leur impact environnemental. Cette documentation présente deux approches principales pour déterminer la taille optimale de la flotte : une méthode basée sur la simulation et une méthode basée sur l'analyse des données historiques.

## 2. Aperçu des méthodes de sizing

- **Méthode basée sur la simulation** : Utilisée lorsque les données historiques sont limitées ou lorsqu'on souhaite explorer un scénario basé sur des probabilités.
- **Méthode basée sur l'analyse des données historiques** : Préférée lorsqu'on dispose de données détaillées sur les expéditions.

## 3. Méthode basée sur la simulation (sans import Excel)

### 3.1 Principes de la méthode

Cette approche utilise des techniques de simulation Monte Carlo pour modéliser le comportement du système logistique sur une période donnée, en se basant sur des paramètres statistiques estimés.

### 3.2 Paramètres d'entrée

- Nombre de jours pour la simulation
- Nombre de simulations à exécuter
- Moyenne et écart-type du nombre d'expéditions par jour
- Moyenne et écart-type du délai de retour des emballages
- Seuil de confiance pour l'estimation de la flotte


### 3.3 Processus de simulation

1. Génération de scénarios de demande et de retour basés sur les distributions spécifiées
2. Calcul du stock nécessaire pour chaque scénario
3. Analyse statistique des résultats pour déterminer la taille de flotte optimale

### 3.4 Interprétation des résultats

- Fonction de répartition (CDF) du stock nécessaire. Il montre la répartition des quantités de stock nécéssaires. Par exemple un seuil de confiance de 99% signifie que le stock recommandé sera suffisant pour répondre à la demande dans 99% des cas simulés.
- Analyse des scénarios (minimum, moyen, maximum)
- Recommandation de taille de flotte basée sur le seuil de confiance

### 3.5 Avantages et limites

**Avantages** :
- Flexible pour explorer différents scénarios
- Ne nécessite pas de données historiques détaillées
- Permet d'évaluer la robustesse de la solution

**Limites** :
- Dépend de la précision des paramètres d'entrée estimés
- Peut ne pas capturer certaines subtilités du système réel

## 4. Méthode basée sur l'analyse des données historiques (avec import Excel)

### 4.1 Principes de la méthode

Cette approche utilise des données historiques détaillées pour analyser les tendances passées et prédire les besoins futurs en emballages.

### 4.2 Import et prétraitement des données

- Import des données depuis un fichier Excel ou CSV
- Nettoyage et formatage des données (gestion des valeurs manquantes, conversion des dates, etc.)
- Visualisation initiale pour détecter les anomalies ou tendances évidentes

### 4.3 Analyse et modélisation

- Utilisation du modèle NeuralProphet pour l'analyse des séries temporelles
- Décomposition des tendances (saisonnalité, tendance générale, effets des jours fériés)
- Évaluation de la qualité du modèle (métriques de performance, analyse des résidus)

### 4.4 Estimation de la flotte

- Génération de prévisions de demande future
- Simulation du système logistique basée sur ces prévisions
- Calcul de la taille de flotte optimale en fonction des contraintes spécifiées

### 4.5 Avantages et limites

**Avantages** :
- Basée sur des données réelles, capturant les subtilités du système
- Peut fournir des insights détaillés sur les tendances et motifs
- Potentiellement plus précise pour des prévisions à court et moyen terme

**Limites** :
- Nécessite des données historiques complètes et de qualité
- Peut être moins adaptée pour des changements radicaux dans les opérations futures

## 5. Choix de la méthode appropriée

Le choix entre les deux méthodes dépend de plusieurs facteurs :

- **Disponibilité des données** : Si vous disposez de données historiques détaillées et fiables, la méthode basée sur l'analyse des données est généralement préférable.
- **Changements anticipés** : Si vous prévoyez des changements significatifs dans vos opérations, la méthode de simulation peut être plus adaptée pour explorer différents scénarios.
- **Précision requise** : Pour une estimation rapide et approximative, la méthode de simulation peut suffire. Pour une analyse plus approfondie, l'analyse des données historiques est recommandée.
- **Ressources disponibles** : L'analyse des données historiques peut nécessiter plus de temps et d'expertise pour l'interprétation des résultats.

## 6. Optimisation des coûts

Les deux méthodes permettent d'intégrer des considérations de coûts :

- **Option 1 : Coût basé sur les points de collecte**
  - Prend en compte le nombre de destinations, les coûts de récupération, et les coûts d'expédition
- **Option 2 : Coût basé sur le poids**
  - Considère le poids des emballages et les coûts associés au transport

L'outil permet d'analyser comment les différents paramètres de coût influencent la taille optimale de la flotte et les coûts totaux.

## 7. Conclusion

Le choix de la méthode de sizing de flotte dépend du contexte spécifique de chaque entreprise. L'outil offre la flexibilité nécessaire pour s'adapter à différentes situations, que ce soit pour une estimation rapide basée sur des paramètres généraux ou une analyse approfondie utilisant des données historiques détaillées. Dans tous les cas, l'objectif est de fournir une base solide pour la prise de décision, permettant d'optimiser la taille de la flotte d'emballages réutilisables tout en minimisant les coûts et l'impact environnemental.
    """)