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

# --- FUNGSI BARU: ANALISIS VARIABEL DAN SARAN SPESIFIK ---
def analyze_and_suggest(df, final_score):
    """Menganalisis data input dan menghasilkan saran spesifik berdasarkan kelemahan (Low & Medium)."""
    saran_spesifik = []
    
    # --- KELOMPOK A: AKADEMIK DAN KEBIASAAN (Fokus Utama) ---
    
    # 1. Hours_Studied
    if df['Hours_Studied'][0] < 15: # Kritis (Low)
        saran_spesifik.append(f"- **Jam Belajar:** Tingkatkan `Study Hours` (saat ini {df['Hours_Studied'][0]} jam) menjadi minimal 20-25 jam per minggu. Ini adalah faktor pendorong skor terbesar.")
    elif df['Hours_Studied'][0] < 20: # Perlu Peningkatan (Medium)
        saran_spesifik.append(f"- **Jam Belajar:** Jam belajar ({df['Hours_Studied'][0]} jam) sudah baik, namun peningkatkan sedikit (20-22 jam) dapat memberi dorongan signifikan ke skor tertinggi.")
    
    # 2. Attendance
    if df['Attendance'][0] < 90: # Kritis (Low)
        saran_spesifik.append(f"- **Kehadiran:** Kehadiran rendah ({df['Attendance'][0]}%) berisiko besar. Harus ditingkatkan menjadi >95% untuk memastikan tidak ada materi yang terlewat.")
    elif df['Attendance'][0] < 95: # Perlu Peningkatan (Medium)
        saran_spesifik.append(f"- **Kehadiran:** Kehadiran ({df['Attendance'][0]}%) mendekati optimal. Targetkan 100% untuk meminimalkan risiko kehilangan materi penting.")
        
    # 3. Tutoring_Sessions
    if df['Tutoring_Sessions'][0] == 0: # Kritis (Low)
        saran_spesifik.append("- **Sesi Bimbingan:** Pertimbangkan untuk menambah Sesi Bimbingan (Tutoring Sessions) secara berkala (1-2 kali/bulan) untuk mengatasi kelemahan spesifik.")
    elif df['Tutoring_Sessions'][0] == 1 and final_score < 80: # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Sesi Bimbingan:** Satu sesi per bulan sudah membantu. Jika skor perlu ditingkatkan, tambahkan menjadi 2-3 sesi per bulan.")
        
    # 4. Previous_Scores
    if df['Previous_Scores'][0] < 70: # Kritis (Low)
        saran_spesifik.append(f"- **Nilai Sebelumnya:** Skor sebelumnya ({df['Previous_Scores'][0]}) rendah. Fokus pada penguatan dasar pengetahuan subjek yang telah lalu.")
    elif df['Previous_Scores'][0] < 80: # Perlu Peningkatan (Medium)
        saran_spesifik.append(f"- **Nilai Sebelumnya:** Skor sebelumnya ({df['Previous_Scores'][0]}) berada di level 'Bagus'. Analisis di mana poin hilang untuk mencapai skor >85.")

    # --- KELOMPOK B: LINGKUNGAN DAN DUKUNGAN ---

    # 5. Parental_Involvement
    if df['Parental_Involvement'][0] == 'Low': # Kritis (Low)
        saran_spesifik.append("- **Keterlibatan Ortu:** Keterlibatan Orang Tua Rendah. Sekolah perlu mengkomunikasikan pentingnya pengawasan dan dukungan belajar di rumah.")
    elif df['Parental_Involvement'][0] == 'Medium' and final_score < 85: # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Keterlibatan Ortu:** Keterlibatan Medium sudah baik. Dorong orang tua untuk terlibat lebih spesifik (misalnya, meninjau hasil belajar mingguan) untuk mencapai hasil optimal.")
        
    # 6. Motivation_Level
    if df['Motivation_Level'][0] == 'Low': # Kritis (Low)
        saran_spesifik.append("- **Motivasi:** Tingkat Motivasi Rendah. Lakukan pendekatan konseling atau temukan mentor untuk memicu semangat belajar siswa.")
    elif df['Motivation_Level'][0] == 'Medium' and final_score < 85: # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Motivasi:** Tingkat Motivasi Medium. Bantuan siswa menetapkan tujuan akademik jangka pendek yang jelas untuk menjaga momentum dan fokus.")

    # 7. Access_to_Resources
    if df['Access_to_Resources'][0] == 'Low': # Kritis (Low)
        saran_spesifik.append("- **Akses Sumber Daya:** Akses Rendah ke buku/alat. Sekolah atau Yayasan perlu menyediakan sumber daya tambahan (misalnya, perpustakaan atau tablet) untuk siswa ini.")
    elif df['Access_to_Resources'][0] == 'Medium': # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Akses Sumber Daya:** Akses Medium. Pastikan siswa tahu cara memaksimalkan sumber daya yang ada dan mempertimbangkan untuk menyediakan beberapa sumber daya premium.")

    # 8. Internet_Access (Hanya ada Yes/No, jadi fokus pada 'No')
    if df['Internet_Access'][0] == 'No':
        saran_spesifik.append("- **Akses Internet:** Ketiadaan akses Internet. Sarankan penggunaan fasilitas sekolah atau perpustakaan untuk pekerjaan rumah dan riset.")
    
    # 9. Family_Income
    if df['Family_Income'][0] == 'Low': # Kritis (Low)
        saran_spesifik.append("- **Pendapatan Keluarga:** Pendapatan Rendah. Pertimbangkan beasiswa atau bantuan dana untuk meringankan beban dan memungkinkan siswa fokus belajar.")
    elif df['Family_Income'][0] == 'Medium' and final_score < 85: # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Pendapatan Keluarga:** Pendapatan Medium. Pastikan tidak ada tekanan finansial tak terduga yang mengganggu fokus belajar siswa.")
        
    # 10. Teacher_Quality
    if df['Teacher_Quality'][0] == 'Low': # Kritis (Low)
        saran_spesifik.append("- **Kualitas Guru:** Nilai kualitas Guru yang rendah. Perlu evaluasi atau pertimbangan untuk memindahkan siswa ke kelas dengan metode pengajaran yang lebih efektif.")
    elif df['Teacher_Quality'][0] == 'Medium' and final_score < 85: # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Kualitas Guru:** Kualitas Guru Medium. Dorong siswa untuk proaktif mencari bantuan tambahan atau materi dari guru lain/mentor.")

    # 11. Peer_Influence
    if df['Peer_Influence'][0] == 'Negative': # Kritis (Low)
        saran_spesifik.append("- **Pengaruh Teman Sebaya:** Pengaruh Negatif. Konseling diperlukan untuk membantu siswa memilih lingkungan pertemanan yang mendukung akademik.")
    elif df['Peer_Influence'][0] == 'Neutral' and final_score < 80: # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Pengaruh Teman Sebaya:** Pengaruh Netral. Dorong siswa untuk aktif berinteraksi dengan teman sebaya yang memiliki tujuan akademik yang positif.")

    # 12. Parental_Education_Level (Jika skor rendah dan pendidikan ortu rendah, perlu intervensi sekolah)
    if df['Parental_Education_Level'][0] == 'High School' and final_score < 75:
        saran_spesifik.append("- **Edukasi Ortu:** Sekolah dapat menawarkan lokakarya bimbingan studi yang ditargetkan untuk orang tua dengan tingkat pendidikan SMA ke bawah.")
    
    # 13. Distance_from_Home
    if df['Distance_from_Home'][0] == 'Far': # Kritis (Low)
        saran_spesifik.append("- **Jarak Rumah:** Jarak yang Jauh dapat memengaruhi energi dan kehadiran. Pastikan transportasi siswa konsisten dan aman.")
    elif df['Distance_from_Home'][0] == 'Moderate' and final_score < 75: # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Jarak Rumah:** Jarak Moderate. Pastikan siswa memiliki waktu istirahat yang cukup setelah bepergian jauh untuk mengembalikan energi belajar.")
        
    # --- KELOMPOK C: KESEHATAN DAN KESEIMBANGAN ---
    
    # 14. Sleep_Hours
    sleep = df['Sleep_Hours'][0]
    if sleep < 6: # Kritis (Low)
        saran_spesifik.append(f"- **Waktu Tidur:** Waktu Tidur terlalu singkat ({sleep} jam). Pastikan siswa tidur 7-8 jam per hari untuk konsentrasi optimal.")
    elif sleep == 6: # Perlu Peningkatan (Medium)
        saran_spesifik.append(f"- **Waktu Tidur:** Waktu Tidur ({sleep} jam) berada di batas minimal. Targetkan 7-8 jam untuk meningkatkan daya ingat dan fokus secara signifikan.")
    elif sleep > 9: # Kritis (Over)
        saran_spesifik.append(f"- **Waktu Tidur:** Waktu Tidur ({sleep} jam) mungkin berlebihan. Perlu evaluasi apakah ini disebabkan oleh kelelahan atau masalah medis.")
        
    # 15. Physical_Activity
    if df['Physical_Activity'][0] < 2: # Kritis (Low)
        saran_spesifik.append("- **Aktivitas Fisik:** Tingkat Aktivitas Fisik rendah. Dorong siswa untuk berolahraga minimal 3 jam per minggu untuk menjaga kesehatan mental dan fokus.")
    elif df['Physical_Activity'][0] < 3: # Perlu Peningkatan (Medium)
        saran_spesifik.append("- **Aktivitas Fisik:** Tingkat Aktivitas Fisik sudah baik. Tambahkan sedikit aktivitas (misalnya, 3-4 jam) untuk manfaat kesehatan mental yang optimal.")
        
    # 16. Extracurricular_Activities
    if df['Extracurricular_Activities'][0] == 'No' and final_score > 70:
        saran_spesifik.append("- **Kegiatan Ekstrakurikuler:** Pertimbangkan untuk mendorong siswa ikut ekstrakurikuler (saat ini 'No') untuk mengembangkan *soft skills* dan manajemen waktu.")

    # 17. Learning_Disabilities
    if df['Learning_Disabilities'][0] == 'Yes':
        saran_spesifik.append("- **Kesulitan Belajar:** Karena siswa memiliki 'Learning Disabilities', intervensi harus melibatkan dukungan guru pendamping dan metode pengajaran adaptif.")

    # 18. School_Type & 19. Gender tidak memiliki saran spesifik yang bersifat intervensi langsung, tetapi dapat dimasukkan dalam evaluasi sekolah secara internal.
        
    return saran_spesifik

# --- 3. JUDUL & HEADER ---
st.title("🎓 Student Exam Score Prediction (LightGBM Model)")
st.markdown("""
This application uses the highly accurate LightGBM Model (MAE 0.76) to predict students' exam scores, taking into account 19 factors (Academic, Environmental, and Personal).
""")

if model is None or preprocessor is None:
    # Menghentikan eksekusi jika file tidak dimuat
    st.stop()

# --- 4. FULL INPUT FORM (19 FEATURES) ---
with st.form("full_prediction_form"):

    # --- SECTION 1: Academic factors ---
    st.subheader("📚 Academic factors") 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Peningkatan default untuk Hours_Studied dari 20 menjadi 30
        hours_studied = st.number_input("Study Hours (per week)", 0, 50, 30, help="Total self-study hours outside school. Ideal 30-40 jam.") 
        # Peningkatan default untuk Attendance dari 80 menjadi 95
        attendance = st.slider("Attendance (%)", 50, 100, 95) 
    
    with col2:
        # Peningkatan default untuk Previous_Scores dari 75 menjadi 90
        previous_scores = st.number_input("Previous Exam Score", 0, 100, 90, help="Nilai rata-rata ujian sebelumnya. Ideal > 90.") 
        tutoring_sessions = st.number_input("Tutoring Sessions (per month)", 0, 10, 1)
        
    with col3:
        # Peningkatan default untuk Sleep_Hours dari 7 menjadi 8
        sleep_hours = st.slider("Sleep Hours (per day)", 4, 10, 8) 
        physical_activity = st.slider("Physical Activity (hours/week)", 0, 10, 3)

    st.divider()

    # --- SECTION 2: Environmental factors ---
    st.subheader("🏠 Environmental factors") 
    
    with st.expander("Fill in Student Environment Details", expanded=True): 
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Peningkatan default dari Medium menjadi High
            parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=2) 
            # Peningkatan default dari Medium menjadi High
            access_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=2) 
            # Peningkatan default dari Medium menjadi High
            family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=2) 
            
        with col5:
            # Peningkatan default dari Medium menjadi High
            teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=2) 
            parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=2)
            peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"], index=0)
            
        with col6:
            school_type = st.selectbox("School Type", ["Public", "Private"])
            internet_access = st.radio("Internet Access", ["Yes", "No"], horizontal=True)
            distance_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

    # --- SECTION 3: OTHER PERSONAL FACTORS ---
    st.subheader("👤 Other Personal Factors") 
    col7, col8, col9 = st.columns(3)
    with col7:
        # Peningkatan default dari Medium menjadi High
        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=2)
    with col8:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    with col9:
        extra_activities = st.checkbox("Participate in Extracurriculars?", value=True)
        learning_disabilities = st.checkbox("Has Learning Disabilities?", value=False)

    # Convert Checkbox to Data Format ("Yes"/"No")
    extra_val = "Yes" if extra_activities else "No"
    learning_val = "Yes" if learning_disabilities else "No"

    submit_btn = st.form_submit_button("🔍 Analysis & Prediction", type="primary") 

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
        st.markdown("### 📊 Model Analysis Results")
        
        res_col1, res_col2 = st.columns([1, 3])
        
        with res_col1:
            st.metric(label="Exam Score Prediction", value=f"{final_score:.2f}")

        # 5. GENERATE KESIMPULAN & SARAN BERDASARKAN SKOR
        saran_spesifik = analyze_and_suggest(input_df, final_score)
        
        if final_score >= 85:
            st.success("🌟 **Category: Very Good.** Students are predicted to achieve outstanding academic performance.")
            conclusion = "The student demonstrates a very strong profile. Needs to maintain consistency and continue to seek opportunities for excellence."
        
        elif final_score >= 70:
            st.info("✅ **Category: Good.** The student is on a satisfactory track but has potential to improve performance.")
            conclusion = "The student has a solid academic foundation, but there are several habitual or environmental factors that could be improved to achieve the 'Very Good' category."

        else: # final_score < 70
            st.error("⚠️ **Category: Intervention Required (Early Warning).** Predicted score is low; immediate action is required.")
            conclusion = "Low predicted scores indicate that students face significant challenges. Primary focus should be given to key factors assessed as weak in the input profile."
        
        with res_col2:
            st.markdown(conclusion)
        
        st.markdown("---")
        st.markdown("### 📋 Specific Action Recommendations")

        if saran_spesifik:
            # Jika ada saran spesifik berdasarkan variabel input yang lemah
            st.warning("Based on student input data, the following are priority actions to improve scores.:")
            for saran in saran_spesifik:
                st.markdown(saran)
        else:
            # Jika skor sangat tinggi dan tidak ada variabel yang 'lemah'
            st.success("All key factors are at optimal levels. No urgent intervention is required other than maintaining consistency.")
        
        # Optional: Display input data for verification
        with st.expander("View Details of Analyzed Student Data"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"A system error occurred during prediction.: {e}")
        st.write("Ensure preprocessor and model objects (.pkl) are valid. Error detail: " + str(e))
