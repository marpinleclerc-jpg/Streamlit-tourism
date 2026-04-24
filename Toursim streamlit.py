import streamlit as st
import pandas as pd
from transformers import pipeline
from google_play_scraper import Sort, reviews
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisa Sentimen Tiket.com", layout="wide")

# --- CACHING MODEL (Agar tidak reload terus menerus) ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="mdhugol/indonesia-bert-sentiment-classification")

@st.cache_data
def scrape_data(app_id, count):
    res, _ = reviews(app_id, lang='id', country='id', sort=Sort.NEWEST, count=count)
    return pd.DataFrame(res)

# --- UI STREAMLIT ---
st.title("📊 Dashboard Analisa Sentimen App Review")
st.subheader("Aplikasi: Tiket.com (Google Play Store)")

# Sidebar untuk input
with st.sidebar:
    st.header("Pengaturan")
    jumlah_review = st.slider("Pilih jumlah review yang akan diambil:", 5, 100, 20)
    proses_btn = st.button("Mulai Analisa")

if proses_btn:
    model_sentimen = load_model()
    
    with st.spinner('Sedang mengambil data dan menganalisa...'):
        # 1. Scrape Data
        df = scrape_data('com.tiket.gits', jumlah_review)
        teks_review = df['content'].tolist()
        
        # 2. Prediksi Sentimen
        hasil_ai = model_sentimen(teks_review)
        
        # 3. Mapping Label
        kamus = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
        df['Sentimen'] = [kamus.get(x['label']) for x in hasil_ai]
        df['Skor'] = [x['score'] for x in hasil_ai]

        # --- TAMPILAN VISUALISASI ---
        col1, col2 = st.columns(2)
        
        hitung_sentimen = df['Sentimen'].value_counts()
        warna = {'positive': '#28a745', 'neutral': '#6c757d', 'negative': '#dc3545'}

        with col1:
            st.write("### Persentase Sentimen")
            fig1, ax1 = plt.subplots()
            ax1.pie(hitung_sentimen, labels=hitung_sentimen.index, autopct='%1.1f%%', 
                    colors=[warna.get(x) for x in hitung_sentimen.index], startangle=140)
            st.pyplot(fig1)

        with col2:
            st.write("### Jumlah Review")
            fig2, ax2 = plt.subplots()
            sns.barplot(x=hitung_sentimen.index, y=hitung_sentimen.values, palette=warna, ax=ax2)
            st.pyplot(fig2)

        # --- TABEL DATA ---
        st.write("### Data Review Terbaru")
        st.dataframe(df[['at', 'userName', 'content', 'Sentimen', 'Skor']].head(10))
else:
    st.info("Silakan tentukan jumlah review di sidebar dan klik 'Mulai Analisa'")