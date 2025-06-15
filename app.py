import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler

# =================================================================================================
# BAGIAN 1: SETUP (Diambil 100% dari kode asli Anda)
# =================================================================================================
url = "https://raw.githubusercontent.com/far1228/Dashboard-DM/refs/heads/main/file.csv"
df = pd.read_csv(url)

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; border: 1px solid #e6e9ef; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .section-header { background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎯 Customer Segmentation Dashboard</h1>
    <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 1.2rem;">Advanced Clustering Analysis with Modern Visualization</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔧 Dashboard Controls")
    uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')
    st.markdown('<div class="section-header"><h3>📊 Dataset Overview</h3></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Records", f"{len(df):,}")
    with col2: st.metric("Features", f"{len(df.columns)}")
    with col3: st.metric("Missing Values", f"{df.isnull().sum().sum()}")
    with col4: st.metric("Data Types", f"{len(df.dtypes.unique())}")
    with st.expander("🔍 View Dataset Sample", expanded=False): st.dataframe(df.head(10), use_container_width=True)
    
    df_clean = df.copy()
    drop_columns = ['Unnamed: 0', 'Transaction_ID', 'Transaction_Date', 'Product_SKU', 'Product_Description', 'Coupon_Code']
    df_clean.drop(columns=drop_columns, inplace=True, errors='ignore')
    num_cols = df_clean.select_dtypes(include='number').columns.tolist()
    cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])
    df_clean[cat_cols] = imputer_cat.fit_transform(df_clean[cat_cols])
    st.success("✅ Data preprocessing completed successfully!")

    with st.sidebar:
        st.markdown("### 🎯 Feature Selection")
        feature_option = st.radio("Select Features:", ("All Features", "Subset Features"))

    if feature_option == "All Features":
        categorical_cols = ['Gender', 'Location', 'Product_Category', 'Coupon_Status', 'Month']
        numerical_cols = ['Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges', 'GST', 'Offline_Spend', 'Online_Spend', 'Discount_pct']
    else:
        categorical_cols = ['Gender', 'Location', 'Product_Category', 'Coupon_Status', 'Month']
        numerical_cols = ['Tenure_Months', 'Quantity', 'Avg_Price', 'Offline_Spend', 'Online_Spend', 'Discount_pct']

    X = df_clean[categorical_cols + numerical_cols]
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('encoder', OneHotEncoder(drop='first', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
    X_processed = preprocessor.fit_transform(X)

    # =================================================================================================
    # BAGIAN 2: KONTROL SIDEBAR DAN LOGIKA UTAMA
    # =================================================================================================
    
    with st.sidebar:
        st.markdown("### 🎨 Clustering Method")
        method = st.selectbox("Choose Algorithm:", ["KMeans", "DBSCAN"])
        
        compare_k_mode = False
        k_for_detailed_analysis = 3 
        
        if method == "KMeans":
            st.markdown("---")
            compare_k_mode = st.checkbox("Bandingkan beberapa nilai K?", help="Menjalankan analisis lengkap berdampingan untuk setiap K yang direkomendasikan.")
            if not compare_k_mode:
                k_for_detailed_analysis = st.slider("🎯 Number of Clusters (k)", 2, 10, 3, key="k_slider_manual")
            else:
                st.info("Mode perbandingan aktif. Pilihan K akan ditentukan secara otomatis.")
        elif method == "DBSCAN":
              eps = st.slider("📏 Epsilon (eps)", 0.1, 10.0, 1.5, step=0.1)
              min_samples = st.slider("👥 Minimum Samples", 2, 50, 10)

    # --- FUNGSI UNTUK MENJALANKAN ANALISIS MENDALAM LENGKAP ---
    def run_complete_analysis_for_k(k, X, df_orig, preprocessor_obj, numerical_cols, categorical_cols, key_prefix=""):
        st.markdown(f"#### Hasil Lengkap untuk K = {k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
        df_result = df_orig.copy()
        df_result['Cluster'] = kmeans.labels_

        # Deteksi Outlier berdasarkan jarak ke centroid
        distances = kmeans.transform(X)
        min_distances = distances[np.arange(len(distances)), kmeans.labels_]
        Q1, Q3 = np.percentile(min_distances, [25, 75])
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR
        df_result['Outlier_Status'] = np.where(min_distances > outlier_threshold, 'Outlier', 'Non-Outlier')
        n_outliers = (df_result['Outlier_Status'] == 'Outlier').sum()

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, kmeans.labels_)
        dbi = davies_bouldin_score(X, kmeans.labels_)
        ch_score = calinski_harabasz_score(X, kmeans.labels_)
        
        cols_metrics = st.columns(4)
        cols_metrics[0].metric("Silhouette", f"{silhouette:.4f}")
        cols_metrics[1].metric("DBI Score", f"{dbi:.4f}")
        cols_metrics[2].metric("Outliers", f"{n_outliers}")
        cols_metrics[3].metric("Inertia", f"{inertia:.1f}")
        
        with st.expander("📈 Detailed Silhouette Analysis", expanded=False):
            sample_values = silhouette_samples(X, kmeans.labels_)
            fig = go.Figure()
            y_lower = 10
            for i in range(k):
                ith_values = sample_values[kmeans.labels_ == i]
                ith_values.sort()
                size_i = ith_values.shape[0]
                y_upper = y_lower + size_i
                fig.add_trace(go.Scatter(x=ith_values, y=list(range(y_lower, y_upper)), mode='lines', fill='tozeroy', name=f'C {i}', line=dict(width=0), fillcolor=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]))
                y_lower = y_upper + 10
            fig.add_vline(x=silhouette, line=dict(color="red", dash="dash", width=2), annotation_text=f"Avg: {silhouette:.3f}")
            st.plotly_chart(fig.update_layout(height=300, title="Silhouette Plot"), use_container_width=True)

        with st.expander("🔬 Analisis Dampak Outlier", expanded=False):
            st.plotly_chart(px.box(df_result, x='Outlier_Status', y=st.selectbox("Pilih Fitur:", numerical_cols, key=f"outlier_vis_{key_prefix}"), color='Outlier_Status', title="Distribusi Fitur vs Status Outlier", color_discrete_map={'Outlier': '#EF553B', 'Non-Outlier': '#636EFA'}), use_container_width=True)
            if n_outliers > 0 and n_outliers < len(df_result):
                df_no_outliers = df_result[df_result['Outlier_Status'] == 'Non-Outlier'].copy()
                X_no_outliers = preprocessor_obj.transform(df_no_outliers[categorical_cols + numerical_cols]) # Gunakan transform, bukan fit_transform
                if len(df_no_outliers) > k: # Pastikan data cukup untuk di-cluster
                    kmeans_no_outliers = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_no_outliers)
                    sil_new = silhouette_score(X_no_outliers, kmeans_no_outliers.labels_)
                    dbi_new = davies_bouldin_score(X_no_outliers, kmeans_no_outliers.labels_)
                    c1,c2 = st.columns(2)
                    c1.markdown("*Metrik Asli (Dengan Outlier)*")
                    c1.metric("Silhouette",f"{silhouette:.4f}")
                    c1.metric("DBI Score",f"{dbi:.4f}")
                    c2.markdown("*Metrik Baru (Tanpa Outlier)*")
                    c2.metric("Silhouette",f"{sil_new:.4f}", f"{sil_new-silhouette:+.4f}")
                    c2.metric("DBI Score", f"{dbi_new:.4f}", f"{dbi_new-dbi:+.4f}")
                else:
                    st.warning("Tidak cukup data tersisa setelah membuang outlier untuk melakukan analisis ulang.")


        pca = PCA(n_components=2).fit(X)
        pca_result = pca.transform(X)
        df_result['PCA1'] = pca_result[:, 0]
        df_result['PCA2'] = pca_result[:, 1]
        st.plotly_chart(px.scatter(df_result, x='PCA1', y='PCA2', color='Cluster', symbol='Outlier_Status', title="PCA Visualization with Outlier Status", color_discrete_sequence=px.colors.qualitative.Set2).update_layout(height=350), use_container_width=True)
        
        tab1, tab2, tab3 = st.tabs(["Statistik Klaster", "Distribusi Fitur", "Profil Centroid"])
        with tab1: st.dataframe(df_result.groupby('Cluster')[numerical_cols].agg(['mean', 'std', 'count']).round(2), use_container_width=True)
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                num_feat_select = st.selectbox("Pilih Fitur Numerik:", numerical_cols, key=f"num_tab_{key_prefix}")
                st.plotly_chart(px.box(df_result, x='Cluster', y=num_feat_select, color='Cluster', title=f"Distribusi {num_feat_select}"), use_container_width=True)
            with col2:
                cat_feat_select = st.selectbox("Pilih Fitur Kategorikal:", categorical_cols, key=f"cat_tab_{key_prefix}" )
                cat_data = df_result.groupby(['Cluster', cat_feat_select]).size().reset_index(name='count')
                st.plotly_chart(px.bar(cat_data, x=cat_feat_select, y='count', color='Cluster', barmode='group', title=f"Distribusi {cat_feat_select}"), use_container_width=True)
        with tab3:
            st.markdown("##### Profil Centroid (Nilai Rata-rata Fitur per Klaster)")
            centroids_scaled = kmeans.cluster_centers_[:, :len(numerical_cols)]
            scaler = preprocessor_obj.named_transformers_['num'].named_steps['scaler']
            centroids_orig = scaler.inverse_transform(centroids_scaled)
            st.dataframe(pd.DataFrame(centroids_orig, columns=numerical_cols).round(2), use_container_width=True)
        
        st.download_button(f"📥 Download Hasil K={k}", df_result.to_csv(index=False), f"kmeans_k{k}results.csv", "text/csv", key=f"dl{key_prefix}")

    # --- KMEANS ---
    if method == "KMeans":
        st.markdown('<div class="section-header"><h3>Analisis KMeans</h3></div>', unsafe_allow_html=True)
        with st.spinner("Menghitung data Elbow Method..."):
            inertias = []
            k_range = range(2, 11)
            for i in k_range:
                kmeans_temp = KMeans(n_clusters=i, random_state=42, n_init='auto').fit(X_processed)
                inertias.append(kmeans_temp.inertia_)
        
        fig_elbow = go.Figure(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers', name='Inertia'))
        st.plotly_chart(fig_elbow.update_layout(title="🔍 Elbow Method Analysis", xaxis_title="Jumlah Klaster (k)", yaxis_title="Inertia"), use_container_width=True)

        if compare_k_mode:
            # Cari K yang direkomendasikan secara matematis
            scaler = MinMaxScaler()
            norm_inertias = scaler.fit_transform(np.array(inertias).reshape(-1, 1))
            line_start = np.array([k_range[0], norm_inertias[0][0]])
            line_end = np.array([k_range[-1], norm_inertias[-1][0]])
            line_vec = line_end - line_start
            
            distances = []
            for i in range(len(k_range)):
                point = np.array([k_range[i], norm_inertias[i][0]])
                dist = np.linalg.norm(np.cross(line_vec, point - line_start)) / np.linalg.norm(line_vec)
                distances.append(dist)
                
            recommended_k = k_range[np.argmax(distances)]
            
            # Ambil K rekomendasi, K-1, dan K+1 (jika valid)
            k_to_run = sorted(list(set([k for k in [recommended_k - 1, recommended_k, recommended_k + 1] if k in k_range])))
            
            st.success(f"📈 *Mode Perbandingan Aktif.* Menjalankan analisis lengkap berdampingan untuk K = *{', '.join(map(str, k_to_run))}* (K={recommended_k} direkomendasikan).")
            st.markdown("---")
            
            cols = st.columns(len(k_to_run))
            for i, k_val in enumerate(k_to_run):
                with cols[i]:
                    run_complete_analysis_for_k(k_val, X_processed, df_clean, preprocessor, numerical_cols, categorical_cols, key_prefix=f"k_{k_val}")
        else:
            # Jalankan analisis untuk K manual
            k = k_for_detailed_analysis
            run_complete_analysis_for_k(k, X_processed, df_clean, preprocessor, numerical_cols, categorical_cols, key_prefix="manual")

    # --- DBSCAN ---
    elif method == "DBSCAN":
        st.markdown('<div class="section-header"><h3>🔍 DBSCAN Clustering Analysis</h3></div>', unsafe_allow_html=True)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X_processed)
        df_result = df_clean.copy()
        df_result['Cluster'] = dbscan.labels_
        
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        n_noise = list(dbscan.labels_).count(-1)
        
        st.markdown("#### *Ringkasan Performa & Data*")
        cols = st.columns(5)
        cols[0].metric("🎯 Clusters Ditemukan", f"{n_clusters}")
        cols[1].metric("🔍 Titik Noise", f"{n_noise}")
        
        if n_clusters > 1:
            valid_points_mask = dbscan.labels_ != -1
            valid_labels = dbscan.labels_[valid_points_mask]
            valid_data = X_processed[valid_points_mask]

            silhouette = silhouette_score(valid_data, valid_labels)
            dbi = davies_bouldin_score(valid_data, valid_labels)
            ch_score = calinski_harabasz_score(valid_data, valid_labels)
            cols[2].metric("📊 Silhouette Score", f"{silhouette:.4f}")
            cols[3].metric("📉 DBI Score", f"{dbi:.4f}")
            cols[4].metric("📈 CH Score", f"{ch_score:.2f}")
        else:
            silhouette, dbi, ch_score = None, None, None
            cols[2].metric("📊 Silhouette Score", "N/A")
            cols[3].metric("📉 DBI Score", "N/A")
            cols[4].metric("📈 CH Score", "N/A")

        if n_clusters > 0:
            pca = PCA(n_components=2).fit(X_processed)
            df_result['PCA1'] = pca.transform(X_processed)[:, 0]
            df_result['PCA2'] = pca.transform(X_processed)[:, 1]
            
            # Pisahkan noise untuk visualisasi yang lebih baik
            df_result['Cluster_str'] = df_result['Cluster'].astype(str)
            df_result.loc[df_result['Cluster'] == -1, 'Cluster_str'] = 'Noise (-1)'
            
            st.plotly_chart(px.scatter(df_result, x='PCA1', y='PCA2', color='Cluster_str', title="🔍 DBSCAN PCA Visualization",
                                      color_discrete_map={ "Noise (-1)": "grey" },
                                      category_orders={"Cluster_str": sorted(df_result['Cluster_str'].unique())}
                                      ).update_layout(height=500), use_container_width=True)
            
            tab1, tab2 = st.tabs(["📊 Analisis Klaster", "🎨 Distribusi Fitur"])
            with tab1:
                valid_clusters_df = df_result[df_result['Cluster'] != -1]
                if not valid_clusters_df.empty:
                    st.dataframe(valid_clusters_df.groupby('Cluster')[numerical_cols].agg(['mean', 'std', 'count']).round(2), use_container_width=True)
                else:
                    st.info("Tidak ada klaster valid yang ditemukan untuk dianalisis.")
            with tab2:
                valid_clusters_df = df_result[df_result['Cluster'] != -1]
                if not valid_clusters_df.empty:
                    c1, c2 = st.columns(2)
                    with c1:
                        num_feat_dbscan = st.selectbox("Fitur Numerik:", numerical_cols, key="dbscan_num_tab")
                        st.plotly_chart(px.box(valid_clusters_df, x='Cluster', y=num_feat_dbscan, color='Cluster'), use_container_width=True)
                    with c2:
                        cat_feat_dbscan = st.selectbox("Fitur Kategorikal:", categorical_cols, key="dbscan_cat_tab")
                        cat_data_dbscan = valid_clusters_df.groupby(['Cluster', cat_feat_dbscan]).size().reset_index(name='count')
                        st.plotly_chart(px.bar(cat_data_dbscan, x=cat_feat_dbscan, y='count', color='Cluster', barmode='group'), use_container_width=True)
                else:
                    st.info("Tidak ada klaster valid yang ditemukan untuk dianalisis distribusinya.")
            
            st.download_button("📥 Download Hasil DBSCAN", df_result.to_csv(index=False), "dbscan_clustering_results.csv", "text/csv")
        else:
            st.warning("⚠ Tidak ada klaster yang ditemukan. Coba sesuaikan parameter Epsilon (eps) dan Minimum Samples.")
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: rgba(70, 130, 180, 0.1); border-radius: 15px; margin: 2rem 0;">
        <h3>🚀 Mulai Analisis</h3>
        <p>Silakan unggah file CSV Anda melalui sidebar untuk memulai analisis segmentasi pelanggan!</p>
    </div>
    """, unsafe_allow_html=True)
