import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title = "Student Placement Analytics",
    page_icon  = "🎓",
    layout     = "wide"
)

# ── Load Model ────────────────────────────────────
@st.cache_resource
def load_model():
    with open('models/placement_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Load Data (Session State) ─────────────────────
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv('data/students_cleaned.csv')

df = st.session_state.df

# ── Sidebar Navigation ────────────────────────────
st.sidebar.title("🎓 Student Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home Dashboard",
    "📊 Student Analytics",
    "🎯 Placement Prediction",
    "💡 Skill Gap Analysis",
    "🔧 Admin Panel"
])

# ════════════════════════════════════════════════
# PAGE 1 — HOME DASHBOARD
# ════════════════════════════════════════════════
if page == "🏠 Home Dashboard":
    st.title("🎓 Student Placement Analytics Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    total      = len(df)
    placed     = (df['placement_status'] == 'Placed').sum()
    not_placed = (df['placement_status'] == 'Not Placed').sum()
    avg_cgpa   = df['cgpa'].mean()

    col1.metric("👥 Total Students", total)
    col2.metric("✅ Placed",         placed)
    col3.metric("❌ Not Placed",     not_placed)
    col4.metric("📈 Avg CGPA",       f"{avg_cgpa:.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            df, names='placement_status',
            title='Overall Placement Status',
            color_discrete_sequence=['#2ecc71','#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dept = df.groupby(['department','placement_status']).size().reset_index(name='count')
        fig = px.bar(
            dept, x='department', y='count',
            color='placement_status',
            title='Department-wise Placement',
            barmode='group',
            color_discrete_sequence=['#2ecc71','#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df, x='placement_status', y='cgpa',
            color='placement_status',
            title='CGPA vs Placement',
            color_discrete_sequence=['#2ecc71','#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x='coding_score',
            color='placement_status',
            title='Coding Score Distribution',
            nbins=20,
            color_discrete_sequence=['#2ecc71','#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════
# PAGE 2 — STUDENT ANALYTICS
# ════════════════════════════════════════════════
elif page == "📊 Student Analytics":
    st.title("📊 Student Analytics")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        dept_filter = st.selectbox(
            "Filter by Department",
            ["All"] + list(df['department'].unique())
        )
    with col2:
        status_filter = st.selectbox(
            "Filter by Placement Status",
            ["All", "Placed", "Not Placed"]
        )

    filtered = df.copy()
    if dept_filter != "All":
        filtered = filtered[filtered['department'] == dept_filter]
    if status_filter != "All":
        filtered = filtered[filtered['placement_status'] == status_filter]

    st.markdown(f"**Showing {len(filtered)} students**")
    st.dataframe(filtered.drop(columns=['placement_encoded'],
                 errors='ignore'), use_container_width=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg CGPA",         f"{filtered['cgpa'].mean():.2f}")
    col2.metric("Avg Coding Score", f"{filtered['coding_score'].mean():.1f}")
    col3.metric("Avg Attendance",   f"{filtered['attendance'].mean():.1f}%")

# ════════════════════════════════════════════════
# PAGE 3 — PLACEMENT PREDICTION
# ════════════════════════════════════════════════
elif page == "🎯 Placement Prediction":
    st.title("🎯 Placement Prediction")
    st.markdown("Student details enter பண்ணி placement chance பாரு!")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        cgpa           = st.slider("CGPA",               5.0, 10.0, 7.5, 0.1)
        attendance     = st.slider("Attendance %",        50,  100,  75)
        coding_score   = st.slider("Coding Score",        0,   100,  60)
        aptitude_score = st.slider("Aptitude Score",      0,   100,  60)
    with col2:
        communication  = st.slider("Communication Score", 0,   100,  60)
        projects       = st.slider("Projects Count",      0,   5,    2)
        internships    = st.slider("Internships",         0,   3,    1)

    st.markdown("---")
    if st.button("🔮 Predict Placement", use_container_width=True):
        input_data   = np.array([[cgpa, attendance, coding_score,
                                  aptitude_score, communication,
                                  projects, internships]])
        input_scaled = scaler.transform(input_data)
        probability  = model.predict_proba(input_scaled)[0]
        placed_prob  = probability[1] * 100

        st.markdown("---")
        if placed_prob >= 70:
            st.success(f"## ✅ HIGH Placement Chance: {placed_prob:.1f}%")
        elif placed_prob >= 45:
            st.warning(f"## ⚠️ MEDIUM Placement Chance: {placed_prob:.1f}%")
        else:
            st.error(f"## ❌ LOW Placement Chance: {placed_prob:.1f}%")

        fig = go.Figure(go.Bar(
            x=['Not Placed', 'Placed'],
            y=[probability[0]*100, probability[1]*100],
            marker_color=['#e74c3c', '#2ecc71']
        ))
        fig.update_layout(title='Placement Probability %', yaxis_title='%')
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════
# PAGE 4 — SKILL GAP ANALYSIS
# ════════════════════════════════════════════════
elif page == "💡 Skill Gap Analysis":
    st.title("💡 Skill Gap Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        coding_score   = st.slider("Coding Score",        0, 100, 50)
        aptitude_score = st.slider("Aptitude Score",      0, 100, 50)
        communication  = st.slider("Communication Score", 0, 100, 50)
    with col2:
        cgpa        = st.slider("CGPA",           5.0, 10.0, 7.0, 0.1)
        projects    = st.slider("Projects Count", 0,   5,    1)
        internships = st.slider("Internships",    0,   3,    0)

    st.markdown("---")
    st.subheader("📋 Recommendations")

    recs = []
    if coding_score < 50:
        recs.append("🔴 **Coding:** DSA practice பண்ணு — LeetCode, HackerRank")
    elif coding_score < 70:
        recs.append("🟡 **Coding:** Medium level problems solve பண்ணு")
    else:
        recs.append("🟢 **Coding:** Excellent!")

    if aptitude_score < 50:
        recs.append("🔴 **Aptitude:** Daily practice பண்ணு — IndiaBix")
    elif aptitude_score < 70:
        recs.append("🟡 **Aptitude:** Mock tests எடு")
    else:
        recs.append("🟢 **Aptitude:** Excellent!")

    if communication < 50:
        recs.append("🔴 **Communication:** Mock interviews practice பண்ணு")
    elif communication < 70:
        recs.append("🟡 **Communication:** Group discussions join பண்ணு")
    else:
        recs.append("🟢 **Communication:** Excellent!")

    if projects < 2:
        recs.append("🔴 **Projects:** Minimum 2-3 projects build பண்ணு")
    else:
        recs.append("🟢 **Projects:** Good!")

    if internships == 0:
        recs.append("🔴 **Internship:** Apply பண்ணு — LinkedIn, Internshala")
    else:
        recs.append("🟢 **Internship:** Good!")

    for r in recs:
        st.markdown(r)

# ════════════════════════════════════════════════
# PAGE 5 — ADMIN PANEL
# ════════════════════════════════════════════════
elif page == "🔧 Admin Panel":
    st.title("🔧 Admin Panel")
    st.markdown("---")

    # ── Password Protection ───────────────────
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == "admin123":
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("❌ Wrong password! Try: admin123")
    else:
        st.success("✅ Admin Logged In!")

        if st.button("🚪 Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.markdown("---")

        # ── Upload Dataset ────────────────────
        st.subheader("📂 Upload New Dataset")
        st.info("CSV file-ல இந்த columns இருக்கணும்: student_id, name, department, cgpa, attendance, coding_score, aptitude_score, communication_score, projects_count, internships, placement_status")

        uploaded_file = st.file_uploader(
            "Choose CSV file", type=['csv']
        )

        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)

                # Required columns check
                required = ['student_id','name','department','cgpa',
                           'attendance','coding_score','aptitude_score',
                           'communication_score','projects_count',
                           'internships','placement_status']

                missing = [c for c in required if c not in new_df.columns]

                if missing:
                    st.error(f"❌ Missing columns: {missing}")
                else:
                    # Preview
                    st.subheader("👀 Data Preview")
                    st.dataframe(new_df.head(10), use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Students", len(new_df))
                    col2.metric("Placed",
                        (new_df['placement_status']=='Placed').sum())
                    col3.metric("Not Placed",
                        (new_df['placement_status']=='Not Placed').sum())

                    st.markdown("---")
                    if st.button("✅ Apply This Dataset", use_container_width=True):
                        # Encode பண்ணு
                        new_df['placement_encoded'] = new_df['placement_status'].map(
                            {'Placed': 1, 'Not Placed': 0}
                        )
                        # Save பண்ணு
                        new_df.to_csv('data/students_cleaned.csv', index=False)
                        # Session update பண்ணு
                        st.session_state.df = new_df
                        st.success("✅ Dataset updated! Dashboard refresh ஆச்சு!")
                        st.balloons()

            except Exception as e:
                st.error(f"❌ Error: {e}")

        st.markdown("---")

        # ── Current Dataset Info ──────────────
        st.subheader("📊 Current Dataset Info")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total",      len(df))
        col2.metric("Placed",     (df['placement_status']=='Placed').sum())
        col3.metric("Not Placed", (df['placement_status']=='Not Placed').sum())
        col4.metric("Avg CGPA",   f"{df['cgpa'].mean():.2f}")