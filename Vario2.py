import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances
import io
import base64
from datetime import datetime

# Configuration de la page
st.set_page_config(
    layout="wide", 
    page_title="GeoVar Pro | Analyse Variographique",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une interface moderne
st.markdown("""
<style>
    /* Entête principale */
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #546E7A;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    /* Signature de l'auteur */
    .author-container {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .author-name {
        color: #1E88E5;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .author-title {
        color: #546E7A;
        font-size: 0.9rem;
    }
    /* Carte de section */
    .section-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .section-title {
        color: #1E88E5;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    /* Boutons */
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Footer */
    footer {
        margin-top: 3rem;
        text-align: center;
        color: #78909C;
        font-size: 0.8rem;
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Entête principale
st.markdown('<h1 class="main-header">GeoVar Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyse variographique avancée pour données d\'exploration minière</p>', unsafe_allow_html=True)

# Fonction pour calculer le variogramme expérimental
def calculate_variogram(df, value_col, max_dist, lag_dist, angle_tolerance, 
                        azimuth, dip, bandwidth, use_3d=True):
    # Extraction des coordonnées et valeurs
    coords = df[['X', 'Y', 'Z']].values
    values = df[value_col].values
    
    # Calcul des distances euclidiennes entre tous les points
    distances = euclidean_distances(coords)
    
    # Initialisation des résultats
    lags = np.arange(lag_dist, max_dist + lag_dist, lag_dist)
    gamma = np.zeros(len(lags))
    counts = np.zeros(len(lags))
    
    # Vecteur directionnel selon azimuth et dip (en radians)
    az_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)
    
    dir_vector = np.array([
        np.cos(dip_rad) * np.sin(az_rad),
        np.cos(dip_rad) * np.cos(az_rad),
        np.sin(dip_rad)
    ])
    
    # Pour chaque paire de points
    n = len(coords)
    for i in range(n):
        for j in range(i+1, n):
            if distances[i, j] <= max_dist:
                # Vecteur entre les points
                h_vector = coords[j] - coords[i]
                h_len = distances[i, j]
                
                # Angle entre le vecteur h et la direction spécifiée
                cos_angle = np.dot(h_vector, dir_vector) / (h_len * np.linalg.norm(dir_vector))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angle_deg = np.degrees(angle)
                
                # Projection perpendiculaire pour vérifier le bandwidth
                if use_3d:
                    proj_len = h_len * np.abs(cos_angle)
                    perp_dist = np.sqrt(h_len**2 - proj_len**2)
                else:
                    h_vector_2d = h_vector[:2]
                    dir_vector_2d = dir_vector[:2]
                    cos_angle_2d = np.dot(h_vector_2d, dir_vector_2d) / (np.linalg.norm(h_vector_2d) * np.linalg.norm(dir_vector_2d) + 1e-10)
                    proj_len = np.linalg.norm(h_vector_2d) * np.abs(cos_angle_2d)
                    perp_dist = np.sqrt(np.linalg.norm(h_vector_2d)**2 - proj_len**2)
                
                # Vérification de la tolérance angulaire et du bandwidth
                if (angle_deg <= angle_tolerance or angle_deg >= (180 - angle_tolerance)) and perp_dist <= bandwidth:
                    # Détermination du bin de lag
                    lag_idx = int(h_len / lag_dist) - 1
                    if 0 <= lag_idx < len(lags):
                        # Calcul de la semi-variance
                        sq_diff = (values[i] - values[j])**2
                        gamma[lag_idx] += sq_diff
                        counts[lag_idx] += 1
    
    # Calcul final du variogramme
    valid_idx = counts > 0
    gamma[valid_idx] = gamma[valid_idx] / (2 * counts[valid_idx])
    
    return lags, gamma, counts

# Fonction pour ajuster le modèle de variogramme
def variogram_model(h, nugget, sill, rang, model_type):
    if model_type == "Sphérique":
        gamma = np.zeros_like(h)
        mask = h <= rang
        gamma[mask] = nugget + (sill - nugget) * (1.5 * (h[mask] / rang) - 0.5 * (h[mask] / rang)**3)
        gamma[~mask] = nugget + (sill - nugget)
        return gamma
    elif model_type == "Exponentiel":
        return nugget + (sill - nugget) * (1 - np.exp(-3 * h / rang))
    elif model_type == "Gaussien":
        return nugget + (sill - nugget) * (1 - np.exp(-3 * (h / rang)**2))
    else:  # Linéaire
        gamma = nugget + (sill - nugget) * (h / rang)
        gamma[h > rang] = nugget + (sill - nugget)
        return gamma

# Sidebar - Chargement des données et paramètres
with st.sidebar:
    st.image("https://via.placeholder.com/150x60/1E88E5/FFFFFF?text=GeoVar+Pro", width=150)
    
    st.markdown("### Chargement des données")
    uploaded_file = st.file_uploader("Fichier CSV/Excel de composites", 
                                     type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        with st.expander("Options de visualisation", expanded=True):
            theme = st.selectbox("Thème", ["Clair", "Sombre", "Professionnel"], index=2)
            decimal_precision = st.slider("Précision décimale", 1, 5, 3)
            show_stats = st.checkbox("Afficher les statistiques", value=True)

# Initialisation de l'application
if uploaded_file is not None:
    try:
        # Chargement des données
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Affichage des données
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-title">Données chargées</h2>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            st.markdown(f"<p>Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Configuration des colonnes dans le sidebar
        with st.sidebar:
            st.markdown("### Configuration des colonnes")
            x_col = st.selectbox("Colonne X", df.columns.tolist(), 
                                 index=df.columns.get_indexer(['X', 'x', 'EAST', 'east', 'EASTING', 'easting'])[0] 
                                 if any(x in df.columns for x in ['X', 'x', 'EAST', 'east', 'EASTING', 'easting']) else 0)
            y_col = st.selectbox("Colonne Y", df.columns.tolist(), 
                                 index=df.columns.get_indexer(['Y', 'y', 'NORTH', 'north', 'NORTHING', 'northing'])[0] 
                                 if any(y in df.columns for y in ['Y', 'y', 'NORTH', 'north', 'NORTHING', 'northing']) else 0)
            z_col = st.selectbox("Colonne Z", df.columns.tolist(), 
                                 index=df.columns.get_indexer(['Z', 'z', 'ELEV', 'elev', 'ELEVATION', 'elevation'])[0] 
                                 if any(z in df.columns for z in ['Z', 'z', 'ELEV', 'elev', 'ELEVATION', 'elevation']) else 0)
            value_col = st.selectbox("Variable d'intérêt", 
                                      [col for col in df.columns if col not in [x_col, y_col, z_col]], 
                                      index=0)
        
        # Renommage des colonnes pour standardisation
        df_work = df.copy()
        df_work = df_work.rename(columns={x_col: 'X', y_col: 'Y', z_col: 'Z'})
        
        # Conversion en numérique si nécessaire
        for col in ['X', 'Y', 'Z', value_col]:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
        
        # Suppression des lignes avec valeurs manquantes
        df_work = df_work.dropna(subset=['X', 'Y', 'Z', value_col])
        
        # Statistiques des données
        if show_stats:
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown(f'<h2 class="section-title">Statistiques de {value_col}</h2>', unsafe_allow_html=True)
                
                # Métriques dans une ligne
                cols = st.columns(5)
                with cols[0]:
                    st.metric("Minimum", f"{df_work[value_col].min():.{decimal_precision}f}")
                with cols[1]:
                    st.metric("Maximum", f"{df_work[value_col].max():.{decimal_precision}f}")
                with cols[2]:
                    st.metric("Moyenne", f"{df_work[value_col].mean():.{decimal_precision}f}")
                with cols[3]:
                    st.metric("Écart-type", f"{df_work[value_col].std():.{decimal_precision}f}")
                with cols[4]:
                    st.metric("CV", f"{df_work[value_col].std() / df_work[value_col].mean():.{decimal_precision}f}")
                
                # Histogramme
                hist_fig = go.Figure()
                hist_fig.add_trace(go.Histogram(
                    x=df_work[value_col],
                    nbinsx=30,
                    marker_color='#1E88E5',
                    opacity=0.75
                ))
                hist_fig.update_layout(
                    title=f"Distribution de {value_col}",
                    xaxis_title=value_col,
                    yaxis_title="Fréquence",
                    template="plotly_white" if theme == "Clair" else "plotly_dark" if theme == "Sombre" else "plotly"
                )
                st.plotly_chart(hist_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Paramètres du variogramme dans le sidebar
        with st.sidebar:
            st.markdown("### Paramètres du variogramme")
            max_dist = st.slider("Distance maximale", 
                                min_value=int(max(df_work['X'].max() - df_work['X'].min(), 
                                                 df_work['Y'].max() - df_work['Y'].min(), 
                                                 df_work['Z'].max() - df_work['Z'].min()) * 0.1),
                                max_value=int(max(df_work['X'].max() - df_work['X'].min(), 
                                                 df_work['Y'].max() - df_work['Y'].min(), 
                                                 df_work['Z'].max() - df_work['Z'].min()) * 0.5),
                                value=int(max(df_work['X'].max() - df_work['X'].min(), 
                                             df_work['Y'].max() - df_work['Y'].min(), 
                                             df_work['Z'].max() - df_work['Z'].min()) * 0.3))
            
            lag_dist = st.slider("Distance de lag", 
                                min_value=int(max_dist/50), 
                                max_value=int(max_dist/5),
                                value=int(max_dist/20))
            
            angle_tolerance = st.slider("Tolérance angulaire (°)", 0, 90, 30)
            bandwidth = st.slider("Bandwidth", 0, max_dist//2, max_dist//10)
            
            use_3d = st.checkbox("Calcul 3D", value=True)
            
            st.markdown("### Direction du variogramme")
            azimuth = st.slider("Azimuth (°)", 0, 360, 0)
            dip = st.slider("Pendage (°)", -90, 90, 0)
        
        # Calcul du variogramme
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-title">Analyse Variographique</h2>', unsafe_allow_html=True)
            
            if st.button("Calculer le variogramme", key="calculate_button"):
                with st.spinner("Calcul du variogramme en cours..."):
                    lags, gamma, counts = calculate_variogram(df_work, value_col, max_dist, lag_dist, 
                                                              angle_tolerance, azimuth, dip, bandwidth, use_3d)
                    
                    # Créer deux colonnes pour le graphique et les paramètres
                    vario_col, model_col = st.columns([2, 1])
                    
                    with vario_col:
                        # Visualisation du variogramme avec Plotly
                        fig = go.Figure()
                        
                        # Configuration du style selon le thème
                        if theme == "Sombre":
                            plot_bgcolor = "#1c1c1c"
                            font_color = "white"
                            grid_color = "#333333"
                            template = "plotly_dark"
                        elif theme == "Professionnel":
                            plot_bgcolor = "#f5f5f5"
                            font_color = "#333333"
                            grid_color = "#dddddd"
                            template = "plotly"
                        else:  # Clair
                            plot_bgcolor = "white"
                            font_color = "#333333"
                            grid_color = "#eeeeee"
                            template = "plotly_white"
                            
                        # Ajouter les points du variogramme expérimental
                        fig.add_trace(go.Scatter(
                            x=lags,
                            y=gamma,
                            mode='markers',
                            name='Variogramme expérimental',
                            marker=dict(
                                size=12,
                                color='#1E88E5',
                                line=dict(width=1, color='#0D47A1')
                            )
                        ))
                        
                        # Ajouter les annotations pour le nombre de paires
                        for i, (lag, gam, count) in enumerate(zip(lags, gamma, counts)):
                            if count > 0:
                                fig.add_annotation(
                                    x=lag,
                                    y=gam,
                                    text=str(int(count)),
                                    showarrow=False,
                                    yshift=10,
                                    font=dict(color=font_color)
                                )
                        
                        # Mise en forme du graphique
                        fig.update_layout(
                            title=f"Variogramme de {value_col} (Az: {azimuth}°, Dip: {dip}°)",
                            xaxis_title="Distance (h)",
                            yaxis_title="Semi-variance γ(h)",
                            hovermode="closest",
                            template=template,
                            plot_bgcolor=plot_bgcolor,
                            paper_bgcolor=plot_bgcolor,
                            font=dict(color=font_color),
                            xaxis=dict(
                                gridcolor=grid_color,
                                zerolinecolor=grid_color
                            ),
                            yaxis=dict(
                                gridcolor=grid_color,
                                zerolinecolor=grid_color
                            ),
                            legend=dict(
                                bgcolor="rgba(255,255,255,0.5)" if theme != "Sombre" else "rgba(0,0,0,0.5)",
                                bordercolor=grid_color
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with model_col:
                        st.markdown("### Modélisation")
                        model_type = st.selectbox("Type de modèle", 
                                                ["Sphérique", "Exponentiel", "Gaussien", "Linéaire"],
                                                index=0)
                        
                        valid_gamma = gamma[counts > 0]
                        valid_lags = lags[counts > 0]
                        
                        if len(valid_gamma) > 0:
                            nugget_suggested = 0
                            sill_suggested = np.nanmax(valid_gamma) if np.isfinite(np.nanmax(valid_gamma)) else np.var(df_work[value_col])
                            range_suggested = valid_lags[np.nanargmax(valid_gamma)] if np.isfinite(np.nanmax(valid_gamma)) else max_dist/3
                            
                            # Format plus compact pour les sliders
                            st.markdown("#### Paramètres du modèle")
                            nugget = st.slider("Effet pépite", 0.0, float(sill_suggested*1.5), float(nugget_suggested), step=sill_suggested/100, format=f"%.{decimal_precision}f")
                            sill = st.slider("Palier", float(nugget), float(sill_suggested*2), float(sill_suggested), step=sill_suggested/100, format=f"%.{decimal_precision}f")
                            rang = st.slider("Portée", float(lag_dist), float(max_dist), float(range_suggested), step=lag_dist, format=f"%.{decimal_precision}f")
                            
                            # Calcul du modèle ajusté
                            h_model = np.linspace(0, max_dist, 100)
                            gamma_model = variogram_model(h_model, nugget, sill, rang, model_type)
                            
                            # Ajout du modèle au graphique
                            fig.add_trace(go.Scatter(
                                x=h_model,
                                y=gamma_model,
                                mode='lines',
                                name=f'Modèle {model_type}',
                                line=dict(color='#FF5252', width=3)
                            ))
                            
                            # Mise à jour du graphique
                            vario_col.plotly_chart(fig, use_container_width=True)
                            
                            # Affichage de la formule du modèle
                            st.markdown("#### Équation du modèle")
                            if model_type == "Sphérique":
                                st.latex(r"\gamma(h) = \begin{cases} " + f"{nugget:.{decimal_precision}f}" + r" + " + f"{(sill-nugget):.{decimal_precision}f}" + r" \times \left[ 1.5 \times \frac{h}{" + f"{rang:.{decimal_precision}f}" + r"} - 0.5 \times \left(\frac{h}{" + f"{rang:.{decimal_precision}f}" + r"}\right)^3 \right] & \text{si } h \leq " + f"{rang:.{decimal_precision}f}" + r" \\ " + f"{sill:.{decimal_precision}f}" + r" & \text{si } h > " + f"{rang:.{decimal_precision}f}" + r" \end{cases}")
                            elif model_type == "Exponentiel":
                                st.latex(r"\gamma(h) = " + f"{nugget:.{decimal_precision}f}" + r" + " + f"{(sill-nugget):.{decimal_precision}f}" + r" \times \left[ 1 - \exp\left(-\frac{3 \times h}{" + f"{rang:.{decimal_precision}f}" + r"}\right) \right]")
                            elif model_type == "Gaussien":
                                st.latex(r"\gamma(h) = " + f"{nugget:.{decimal_precision}f}" + r" + " + f"{(sill-nugget):.{decimal_precision}f}" + r" \times \left[ 1 - \exp\left(-3 \times \left(\frac{h}{" + f"{rang:.{decimal_precision}f}" + r"}\right)^2\right) \right]")
                            else:  # Linéaire
                                st.latex(r"\gamma(h) = \begin{cases} " + f"{nugget:.{decimal_precision}f}" + r" + " + f"{(sill-nugget):.{decimal_precision}f}" + r" \times \frac{h}{" + f"{rang:.{decimal_precision}f}" + r"} & \text{si } h \leq " + f"{rang:.{decimal_precision}f}" + r" \\ " + f"{sill:.{decimal_precision}f}" + r" & \text{si } h > " + f"{rang:.{decimal_precision}f}" + r" \end{cases}")
                            
                            # Résumé des paramètres
                            st.markdown("#### Résumé des paramètres")
                            param_df = pd.DataFrame({
                                'Paramètre': ['Effet pépite', 'Palier', 'Portée', 'Modèle'],
                                'Valeur': [f"{nugget:.{decimal_precision}f}", f"{sill:.{decimal_precision}f}", f"{rang:.{decimal_precision}f}", model_type]
                            })
                            st.dataframe(param_df, use_container_width=True, hide_index=True)
                            
                            # Option d'exportation
                            st.markdown("#### Exportation")
                            export_col1, export_col2 = st.columns(2)
                            with export_col1:
                                if st.button("Exporter les résultats"):
                                    # Créer un DataFrame avec le modèle calculé
                                    export_df = pd.DataFrame({
                                        'Distance': h_model,
                                        'Semi-variance_modèle': gamma_model
                                    })
                                    # Ajouter les points expérimentaux
                                    exp_df = pd.DataFrame({
                                        'Distance': lags,
                                        'Semi-variance_exp': gamma,
                                        'Nombre_paires': counts
                                    })
                                    # Convertir en CSV
                                    csv = export_df.to_csv(index=False)
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="variogramme_{value_col}.csv">📥 Télécharger CSV</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                            
                            with export_col2:
                                if st.button("Copier paramètres"):
                                    param_text = f"Modèle: {model_type}\nEffet pépite: {nugget:.{decimal_precision}f}\nPalier: {sill:.{decimal_precision}f}\nPortée: {rang:.{decimal_precision}f}"
                                    st.code(param_text)
                                    st.success("Paramètres copiés dans le presse-papier!")
                        else:
                            st.error("Pas assez de données pour la modélisation du variogramme.")
            else:
                st.info("Configurez les paramètres et cliquez sur 'Calculer le variogramme' pour démarrer l'analyse.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement: {str(e)}")

else:
    # Guide d'utilisation avec interface moderne
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Guide d\'utilisation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Analyser vos données minières en quelques étapes
        
        GeoVar Pro vous permet d'effectuer une analyse variographique complète de vos données d'exploration minière:
        
        1. **Chargez vos données** en utilisant le sélecteur de fichiers dans la barre latérale
        2. **Sélectionnez les colonnes** correspondant aux coordonnées X, Y, Z et à la variable d'intérêt
        3. **Configurez les paramètres du variogramme**:
           - Distance maximale et pas (lag)
           - Tolérance angulaire et bandwidth
           - Direction (azimuth et pendage)
        4. **Calculez le variogramme** en cliquant sur le bouton
        5. **Modélisez le variogramme** en ajustant les paramètres (effet pépite, palier, portée)
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Exemple de données
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Exemple de format de données</h2>', unsafe_allow_html=True)
        
        # Créer un exemple de données
        example_data = pd.DataFrame({
            'X': [100, 110, 120, 130, 140, 150, 160],
            'Y': [200, 210, 220, 230, 240, 250, 260],
            'Z': [10, 11, 12, 13, 14, 15, 16],
            'GOLD': [1.2, 1.5, 0.8, 2.3, 1.7, 1.9, 1.4],
            'SILVER': [5.6, 4.3, 6.7, 3.2, 5.9, 4.8, 6.2],
            'COPPER': [0.5, 0.8, 0.3, 1.2, 0.7, 0.9, 0.6]
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        # Option de téléchargement des données d'exemple
        csv = example_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="exemple_donnees_minieres.csv" style="display: inline-block; padding: 0.5rem 1rem; background-color: #1E88E5; color: white; text-decoration: none; border-radius: 4px; margin-top: 1rem;">📥 Télécharger les données d\'exemple</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Signature de l'auteur
st.markdown('<div class="author-container">', unsafe_allow_html=True)
st.markdown('<p class="author-name">Didier Ouedraogo, P.Geo</p>', unsafe_allow_html=True)
st.markdown('<p class="author-title">Géostatisticien | Expert en Exploration Minière</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Pied de page
st.markdown('<footer>GeoVar Pro © 2025 | Dernière mise à jour: Mars 2025</footer>', unsafe_allow_html=True)