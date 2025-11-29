import streamlit as st
import pandas as pd
import numpy as np
import joblib 

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide" # Use wide layout for better input visibility
)

# --- 2. LOAD MODEL & PREPROCESSOR ---
@st.cache_resource
def load_assets():
    """
    Loads the LightGBM model (.pkl) and the preprocessor (.pkl).
    Catatan: Model final adalah LightGBM dengan MAE 0.76.
    """
    try:
        # Load LightGBM model
        model = joblib.load('final_model.pkl')
        # Load the same preprocessor
        preprocessor = joblib.load('preprocessor.pkl') 
        return model, preprocessor
    except Exception as e:
        # Menampilkan error jika file tidak ditemukan
        st.error(f"Error memuat file model atau preprocessor: {e}")
        st.error("⚠️ Pastikan file 'final_model.pkl' dan 'preprocessor.pkl' sudah diupload.")
        return None, None

model, preprocessor = load_assets()

# --- 3. JUDUL & HEADER ---
st.title("🎓 Prediksi Performa Siswa (Model LightGBM)")
st.markdown("""
Aplikasi ini menggunakan **Model LightGBM** yang sangat akurat (MAE 0.76) untuk memprediksi nilai ujian siswa, mempertimbangkan **19 faktor** (Akademik, Lingkungan, dan Personal).
""")

if model is None or preprocessor is None:
    # Menghentikan eksekusi jika file tidak dimuat
    st.stop()

# --- 4. FULL INPUT FORM (19 FEATURES) ---
with st.form("full_prediction_form"):

    # --- SECTION 1: AKADEMIK & KEBIASAAN ---
    st.subheader("📚 Data Akademik & Kebiasaan") # Judul dalam B. Indonesia
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hours_studied = st.number_input("Study Hours (per week)", 0, 50, 20, help="Total self-study hours outside school.")
        attendance = st.slider("Attendance (%)", 50, 100, 80)
    
    with col2:
        previous_scores = st.number_input("Previous Exam Score", 0, 100, 75, help="Nilai rata-rata ujian sebelumnya.")
        tutoring_sessions = st.number_input("Tutoring Sessions (per month)", 0, 10, 1)
        
    with col3:
        sleep_hours = st.slider("Sleep Hours (per day)", 4, 10, 7)
        physical_activity = st.slider("Physical Activity (hours/week)", 0, 10, 3)

    st.divider()

    # --- SECTION 2: LINGKUNGAN & DUKUNGAN ---
    st.subheader("🏠 Lingkungan & Dukungan") # Judul dalam B. Indonesia
    
    with st.expander("Isi Detail Lingkungan Siswa", expanded=True): # Sub-judul dalam B. Indonesia
        col4, col5, col6 = st.columns(3)
        
        with col4:
            parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
            access_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
            family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
            
        with col5:
            teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
            parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=1)
            peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"], index=1)
            
        with col6:
            school_type = st.selectbox("School Type", ["Public", "Private"])
            internet_access = st.radio("Internet Access", ["Yes", "No"], horizontal=True)
            distance_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

    # --- SECTION 3: FAKTOR PERSONAL LAINNYA ---
    st.subheader("👤 Faktor Personal Lainnya") # Judul dalam B. Indonesia
    col7, col8, col9 = st.columns(3)
    with col7:
        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
    with col8:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    with col9:
        extra_activities = st.checkbox("Participate in Extracurriculars?", value=True)
        learning_disabilities = st.checkbox("Has Learning Disabilities?", value=False)

    # Convert Checkbox to Data Format ("Yes"/"No")
    extra_val = "Yes" if extra_activities else "No"
    learning_val = "Yes" if learning_disabilities else "No"

    submit_btn = st.form_submit_button("🔍 Analisis & Prediksi", type="primary") # Tombol dalam B. Indonesia

# --- 5. PREDICTION LOGIC & OUTPUT ---
if submit_btn:
    # VALIDASI INPUT DASAR
    if previous_scores > 100 or previous_scores < 0:
        st.error("⚠️ Previous Exam Score harus berada di antara 0 dan 100.")
        st.stop()
        
    # Construct data according to the X_train column order (CRUCIAL!)
    input_data = {
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Parental_Involvement': [parental_involvement], 
        'Access_to_Resources': [access_resources],
        'Extracurricular_Activities': [extra_val],
        'Sleep_Hours': [sleep_hours],
        'Previous_Scores': [previous_scores],
        'Motivation_Level': [motivation],
        'Internet_Access': [internet_access],
        'Tutoring_Sessions': [tutoring_sessions],
        'Family_Income': [family_income],
        'Teacher_Quality': [teacher_quality],
        'School_Type': [school_type],
        'Peer_Influence': [peer_influence],
        'Physical_Activity': [physical_activity],
        'Learning_Disabilities': [learning_val],
        'Parental_Education_Level': [parental_education],
        'Distance_from_Home': [distance_home],
        'Gender': [gender]
    }

    # Create DataFrame
    input_df = pd.DataFrame(input_data)

    try:
        # 1. Preprocessing Pipeline
        processed_input = preprocessor.transform(input_df)
        
        # 2. LightGBM Model Prediction
        prediction = model.predict(processed_input)
        
        # Get the first prediction value
        final_score = float(prediction[0])
        
        # 3. Constrain score 0-100 (Wajib karena LGBM bisa memprediksi di luar rentang)
        final_score = max(0.0, min(100.0, final_score))
        
        # 4. Display Output and Recommendation Logic
        st.divider()
        st.markdown("### 📊 Hasil Analisis Model")
        
        res_col1, res_col2 = st.columns([1, 3])
        
        with res_col1:
            st.metric(label="Prediksi Skor Ujian", value=f"{final_score:.2f}")

        # 5. GENERATE KESIMPULAN & SARAN BERDASARKAN SKOR
        kesimpulan = ""
        saran = ""
        
        if final_score >= 85:
            st.success("🌟 **Kategori: Sangat Baik.** Siswa diprediksi mencapai performa akademik yang luar biasa.")
            kesimpulan = "Siswa menunjukkan profil yang kuat di hampir semua faktor kunci. Prediksi skor 85 atau lebih tinggi menunjukkan peluang keberhasilan yang sangat tinggi. Perlu mempertahankan konsistensi."
            saran = "1. **Pertahankan Kebiasaan:** Pastikan Jam Belajar, Kehadiran, dan Keterlibatan Orang Tua tetap optimal. 2. **Eksplorasi Mendalam:** Arahkan siswa untuk mendalami topik yang diminati (misalnya, melalui proyek atau penelitian independen) untuk mencapai keunggulan. 3. **Perencanaan Karir:** Mulai diskusikan pilihan universitas atau jalur karir."

        elif final_score >= 70:
            st.info("✅ **Kategori: Baik.** Siswa berada di jalur yang memuaskan namun memiliki potensi untuk meningkatkan performa.")
            kesimpulan = "Siswa memiliki dasar akademis yang solid. Faktor-faktor lingkungan dan kebiasaan menunjukkan keseimbangan, namun ada satu atau dua area yang menahan skor untuk mencapai kategori 'Sangat Baik'."
            saran = f"1. **Fokus pada Belajar:** Tingkatkan `Study Hours` ke 25-30 jam/minggu. 2. **Maksimalkan Dukungan:** Tambahkan `Tutoring Sessions` atau manfaatkan sumber daya akademik di sekolah. 3. **Cek Keseimbangan:** Pastikan `Sleep Hours` dan `Physical Activity` berada di zona optimal (7-8 jam tidur, 3-5 jam olahraga/minggu) untuk menghindari burnout."
            
        else: # final_score < 70
            st.error("⚠️ **Kategori: Perlu Intervensi (Peringatan Dini).** Skor diprediksi rendah; tindakan segera diperlukan.")
            kesimpulan = "Prediksi skor di bawah 70 mengindikasikan bahwa siswa menghadapi tantangan signifikan. Area `Previous Scores`, `Motivation Level`, atau `Parental Involvement` kemungkinan menjadi faktor utama yang memerlukan perhatian segera."
            saran = f"1. **Tinjau Motivasi & Keterlibatan:** Adakan pertemuan dengan orang tua untuk meningkatkan `Parental Involvement`. Cek `Motivation Level` siswa secara berkala. 2. **Intervensi Akademik Cepat:** Fokus pada subjek terlemah. Pertimbangkan sesi bimbingan tambahan (Tutor) jika `Tutoring Sessions` masih rendah. 3. **Perbaiki Dasar:** Tingkatkan `Attendance` (kehadiran) dan pastikan `Hours_Studied` mencapai minimal 15 jam/minggu."

        with res_col2:
            st.markdown(kesimpulan)
        
        st.markdown("---")
        st.markdown("### 📋 Rekomendasi & Saran Tindakan")
        st.warning(saran)
        
        # Optional: Display input data for verification
        with st.expander("Lihat Detail Data Siswa yang Dianalisis"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Terjadi kesalahan sistem saat prediksi: {e}")
        st.write("Pastikan objek preprocessor dan model (.pkl) valid. Error detail: " + str(e))
