import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Configuration de la page
st.set_page_config(
    page_title="QA Augmentée- Prédiction de Priorité",
    page_icon="",
    layout="wide"
)

# Styles CSS personnalisés
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    text-align: center;
    background: #f0f2f6;
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Chemins des fichiers
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "priority_classifier.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"
DATA_PATH = BASE_DIR / "data" / "issues.csv"

# Chargement des modèles
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Modèles non trouvés. Assurez-vous d’avoir exécuté train_model.py")
        return None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.warning("Dataset non trouvé")
        return None

# Header
st.markdown('<div class="main-header">QA Augmentée-Dashboard de Prédiction</div>', unsafe_allow_html=True)

# Charger les modèles
model, vectorizer = load_models()
df = load_data()

if model is None or vectorizer is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=150)
    st.title("Navigation")
    page = st.radio("Choisir une section", ["Prédiction", "Statistiques", "Performance", "À propos"])
    st.markdown("---")
    st.markdown("### Informations")
    if df is not None:
        st.metric("Issues dans le dataset", len(df))
        st.metric("Features utilisées", "TF-IDF (500)")
        st.metric("Modèle", "Random Forest")

# Page 1: Prédiction
if page == "Prédiction":
    st.header("Testez le Modèle de Prédiction")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Entrez les détails de l’anomalie")
        title = st.text_input("Titre de l’issue", placeholder="Ex: Critical memory leak in login module")
        body = st.text_area("Description détaillée", height=150, placeholder="Ex: The application crashes after 1 hour of usage...")

        if st.button("Prédire la Priorité", type="primary"):
            if title and body:
                text = f"{title} {body}"
                X = vectorizer.transform([text])
                priority = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                confidence = max(probabilities) * 100

                st.markdown("---")
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Priorité Prédite", priority)
                with col_b:
                    st.metric("Confiance", f"{confidence:.1f}%")
                with col_c:
                    emoji = ""  # tu peux mettre un emoji selon la priorité
                    st.metric("Urgence", emoji)

                st.markdown("</div>", unsafe_allow_html=True)

                prob_df = pd.DataFrame({'Priorité': model.classes_, 'Probabilité': probabilities * 100})
                fig = px.bar(prob_df, x='Priorité', y='Probabilité',
                             color='Probabilité', color_continuous_scale='RdYlGn_r',
                             labels={'Probabilité': 'Probabilité (%)'}, text='Probabilité')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Veuillez remplir le titre et la description")

    with col2:
        st.subheader("Exemples")
        st.markdown("""
        **P0-Critique**
        - "Critical security vulnerability"
        - "Data loss on production"
        - "System crash on startup"

        **P1-Élevée**
        - "Performance regression"
        - "Major feature broken"
        - "API error 500"

        **P2-Moyenne**
        - "Minor bug in UI"
        - "Incorrect validation message"

        **P3-Faible**
        - "Typo in documentation"
        - "Feature request: dark mode"
        """)

# Page 2: Statistiques
elif page == "Statistiques":
    st.header("Analyse du Dataset")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Issues", len(df))
        with col2:
            st.metric("P0 (Critiques)", len(df[df['priority'] == 'P0']))
        with col3:
            st.metric("P1 (Élevées)", len(df[df['priority'] == 'P1']))
        with col4:
            st.metric("P2+P3", len(df[df['priority'].isin(['P2','P3'])]))

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution des Priorités")
            priority_counts = df['priority'].value_counts().reset_index()
            priority_counts.columns = ['Priorité', 'Nombre']
            fig = px.pie(priority_counts, values='Nombre', names='Priorité',
                         color_discrete_sequence=px.colors.sequential.RdBu_r, hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Répartition par Priorité")
            fig = px.bar(priority_counts, x='Priorité', y='Nombre',
                         color='Priorité',
                         color_discrete_map={'P0':'#d62728','P1':'#ff7f0e','P2':'#2ca02c','P3':'#1f77b4'})
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Aperçu des Données")
        st.dataframe(df.head(10), use_container_width=True)

# Page 3: Performance
elif page == "Performance":
    st.header("Évaluation du Modèle")
    st.markdown("""
    ### Métriques de Performance
    Le modèle a été évalué sur un ensemble de test représentant 20% des données.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Précision Globale", "75%")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Rappel P2", "100%")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("F1-Score Moyen", "0.69")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Matrice de Confusion")
    confusion_img = BASE_DIR / "models" / "confusion_matrix.png"
    if confusion_img.exists():
        st.image(str(confusion_img), caption="Matrice de confusion du modèle", use_container_width=True)
    else:
        st.info("Exécutez train_model.py pour générer la matrice de confusion")
    st.markdown("---")
    st.subheader("Interprétation")
    st.success("""
    ** Points forts :**
    - Excellence sur la classe P2 (Medium) avec un rappel de 100%
    - Bonne précision globale malgré la taille réduite du dataset
    - Aucune confusion majeure sur les priorités critiques
    """)
    st.warning("""
    ** Axes d’amélioration :**
    - Augmenter le nombre d’exemples pour les classes rares (P0, P3)
    - Intégrer des embeddings BERT pour capturer le contexte sémantique
    - Ajouter des features complémentaires (auteur, historique, labels)
    """)

# Page 4: À propos
else:
    st.header("À Propos du Projet")
    st.markdown("""
    ## QA Augmentée-Classement Automatique des Anomalies
    ### Objectif
    Ce projet démontre comment l’**Intelligence Artificielle** peut révolutionner l’Assurance Qualité en automatisant la classification des anomalies logicielles par priorité.

    ### Technologies Utilisées
    - **Machine Learning** : Random Forest Classifier (Scikit-learn)
    - **Vectorisation** : TF-IDF (500 features, bigrammes)
    - **Visualisation** : Streamlit, Plotly
    - **CI/CD** : GitHub Actions (workflow automatisé)

    ### Dataset
    - Source : GitHub API (Issues publiques)
    - Taille : 40 issues
    - Labels : P0 (Critique), P1 (Élevée), P2 (Moyenne), P3 (Faible)

    ### Déploiement
    Pour lancer ce dashboard localement :
    ```bash
    pip install streamlit plotly
    streamlit run dashboard.py
    ```

    ### Intégrations Possibles
    - Jira API
    - GitLab CI/CD
    - Azure DevOps
    - GitHub Actions (déjà implémenté)

    **Développé dans le cadre du cours d’Assurance Qualité Logicielle**
    École Hassania des Travaux Publics (EHTP)
    """)
st.balloons()
