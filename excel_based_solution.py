import streamlit as st
import pandas as pd
from scipy import stats

def load_and_analyze_data(uploaded_file):
    # Déterminer le type de fichier
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # Charger le fichier en fonction de son type
    if file_type == 'xlsx':
        df = pd.read_excel(uploaded_file, index_col=None)
    elif file_type == 'csv':
        df = pd.read_csv(uploaded_file, index_col=None)
    else:
        st.error("Format de fichier non pris en charge. Veuillez utiliser un fichier Excel (.xlsx) ou CSV.")
        return None
    
    # Vérifier si les colonnes requises existent
    if "Export" not in df.columns or "Retour" not in df.columns:
        st.error("Le fichier doit contenir les colonnes 'Export' et 'Retour'")
        return None
    
    # Calculer les statistiques
    export_mean = df["Export"].mean()
    export_std = df["Export"].std()
    retour_mean = df["Retour"].mean()
    retour_std = df["Retour"].std()
    
    # Effectuer le test de normalité (Shapiro-Wilk)
    _, p_value_export = stats.shapiro(df["Export"])
    _, p_value_retour = stats.shapiro(df["Retour"])
    
    return df, export_mean, export_std, retour_mean, retour_std, p_value_export, p_value_retour

def load_data(uploaded_file):
    if uploaded_file is not None:
        result = load_and_analyze_data(uploaded_file)
            
        if result is not None:
            df, export_mean, export_std, retour_mean, retour_std, p_value_export, p_value_retour = result
            
            st.markdown("### Aperçu du fichier chargé :")
            st.dataframe(df.head())
            
            st.markdown("### Statistiques :")
            st.markdown(f"- **Moyenne des exports :** {export_mean:.2f}")
            st.markdown(f"- **Écart-type des exports :** {export_std:.2f}")
            st.markdown(f"- **Moyenne des retours :** {retour_mean:.2f}")
            st.markdown(f"- **Écart-type des retours :** {retour_std:.2f}")
            
            st.markdown("### Test de normalité (Shapiro-Wilk) :")
            st.markdown(f"- **p-value pour les exports :** {p_value_export:.4f}")
            st.markdown(f"- **p-value pour les retours :** {p_value_retour:.4f}")
            
            alpha = 0.05
            st.markdown(f"Les données suivent une loi normale si la p-value est supérieure à {alpha}")
            
            if p_value_export > alpha:
                st.markdown("- **Les données d'export suivent approximativement une loi normale.**")
            else:
                st.markdown("- **Les données d'export ne suivent pas une loi normale.**")
            
            if p_value_retour > alpha:
                st.markdown("- **Les données de retour suivent approximativement une loi normale.**")
            else:
                st.markdown("- **Les données de retour ne suivent pas une loi normale.**")

            # Retourner les statistiques si nécessaire
            return export_mean, export_std, retour_mean, retour_std
    return None, None, None, None