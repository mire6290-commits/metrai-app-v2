import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(page_title="Metrai AI - Charpente", page_icon="🏗️", layout="wide")

API_URL = "https://amira221-metrai-backend.hf.space"

def reset_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.analyzed = False

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

st.title("🏗️ Metrai AI - Métré Charpente Métallique")
st.markdown("Uploadez votre plan PDF pour générer automatiquement la nomenclature complète.")

# Layout: sidebar for inputs
with st.sidebar:
    st.header("1. Upload du Plan")
    uploaded_file = st.file_uploader("Choisissez un plan PDF", type="pdf", on_change=reset_state)
    
    if uploaded_file is not None:
        btn_placeholder = st.empty()
        clicked = btn_placeholder.button("Lancer l'analyse AI 🚀", use_container_width=True, type="primary")
        
        if clicked:
            btn_placeholder.button("Analyse en cours... Patientez ⏳", disabled=True, use_container_width=True)
            reset_state()
            st.session_state.analyzed = False
            with st.spinner("Analyse du plan en cours par l'IA..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/extract", files=files, timeout=600)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.raw_data = data
                        profiles = data.get("profiles", [])
                        st.session_state.profiles_df = pd.DataFrame(profiles)
                        st.session_state.analyzed = True
                        st.success("Analyse terminée avec succès !")
                    else:
                        st.error(f"Erreur du serveur: {response.text}")
                except Exception as e:
                    st.error(f"Une erreur est survenue: {str(e)}")
                    
    if st.session_state.get("analyzed", False):
        st.divider()
        st.markdown("---")
        if st.button("🔄 Nouvelle Analyse / Réinitialiser", use_container_width=True):
            reset_state()
            st.rerun()

# Main area for results
if st.session_state.get("analyzed", False):
    data = st.session_state.raw_data
    df = st.session_state.profiles_df
    
    st.header("📊 Résultats de l'extraction")
    
    # 1. Infos générales & Métriques
    total_poids = data.get("total_weight_kg", 0.0)
    
    # Fallback si total_weight_kg n'est pas dispo
    if not total_poids and "poids_total_kg" in df.columns:
        total_poids = pd.to_numeric(df["poids_total_kg"], errors='coerce').sum()

    # Calculate Total Peinture
    total_peinture = 0.0
    if "surface_peinture_m2" in df.columns:
        total_peinture = pd.to_numeric(df["surface_peinture_m2"], errors='coerce').sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("⚖️ Poids Total", f"{total_poids:,.2f} kg".replace(',', ' '))
    col2.metric("🎨 Surface Peinture", f"{total_peinture:,.2f} m²".replace(',', ' '))
    col3.metric("📦 Éléments", f"{len(df)} profilés")
    col4.metric("📝 Plan", data.get("drawing_type", "Inconnu").capitalize())
    col5.metric("📄 Pages", data.get("pages_processed", 1))
    
    warnings = data.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(f"⚠️ {w}")
            
    unreadable = data.get("unreadable_zones", [])
    if unreadable:
        st.error(f"🚫 Zones illisibles: {', '.join(unreadable)}")

    st.divider()
    st.subheader("✏️ Éditeur de Nomenclature")
    st.markdown("Le tableau ci-dessous est **interactif**. Vous pouvez double-cliquer sur une cellule pour la modifier (ex: corriger une longueur, une quantité ou une désignation) **avant l'exportation**.")
    
    # 2. Editable DataFrame
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    st.subheader("📥 Exportation Finale")
    st.markdown("Une fois vos modifications terminées, téléchargez le fichier Excel au format Expert. Les modifications apportées dans le tableau seront prises en compte.")
    
    # Bouton d'export Excel via l'API
    # On reconstruit les dictionnaires à partir du dataframe édité
    import numpy as np
    edited_df = edited_df.replace({np.nan: None})
    edited_profiles = edited_df.to_dict(orient="records")
    
    try:
        # On n'utilise pas le spinner ici car Streamlit rechargera le bouton de téléchargement 
        # et il faut que le st.download_button s'affiche immédiatement.
        # Donc on génère l'Excel en mémoire côté client ou via l'API avant de l'injecter.
        export_res = requests.post(f"{API_URL}/export/excel", json={"data": edited_profiles})
        if export_res.status_code == 200:
            st.download_button(
                label="📥 Télécharger l'Excel (Format Expert)",
                data=export_res.content,
                file_name="metrai_expert_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        else:
            st.error("Erreur lors de la génération de l'Excel.")
    except Exception as e:
        st.error(f"Erreur d'export: {str(e)}")
