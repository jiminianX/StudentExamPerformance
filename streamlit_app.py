## Step 00 - Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Student Performance Analysis 🎓",
    layout="centered",
    page_icon="🎓",
)

## Step 01 - Setup
st.sidebar.title("Student Performance Analysis 🎓")
page = st.sidebar.selectbox(
    "Select Page",
    ["Business Case 📘", "Visualization 📊", "Prediction 🤖", "Insights and Recommendations 🧠"]
)

st.write("   ")

@st.cache_data
def load_data(path='StudentPerformanceFactors.csv'):
    df_local = pd.read_csv(path)
    original_count = len(df_local)
    missing_before = df_local.isnull().sum().sum()
    return df_local, original_count, missing_before

(df, original_row_count, missing_before) = load_data()

## Step 02 - Pages
if page == "Business Case 📘":

    st.subheader("Student Performance Factors Dashboard")

    st.image("students-sitting-exams-s.png", use_container_width=True)

    st.markdown("[🔗 View dataset source on Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)")

    st.markdown("""
    ## 🎯 Business Problem

    Student academic performance is shaped by a complex mix of factors:

    - **Study habits** such as hours studied and attendance
    - **Home environment** including parental involvement and family income
    - **School factors** like teacher quality and access to resources
    - **Personal factors** such as motivation, sleep, and learning disabilities

    Understanding these drivers allows educators and policymakers to design targeted interventions.
    """)

    st.markdown("""
    ## Our Solution

    1. **Data Analysis:** Identify key factors influencing exam scores
    2. **Visualization:** Interactive dashboards to explore performance patterns
    3. **Predictive Modeling:** Linear regression to predict exam score from student attributes
    """)

    st.markdown("""
    ## Why This Matters

    - **Early intervention:** Identifying at-risk students early can significantly improve outcomes.
    - **Resource allocation:** Schools can direct tutoring and support to students who need it most.
    - **Policy design:** Evidence-based strategies improve system-wide performance.
    """)

    with st.expander("📘 Data Dictionary (click to expand)"):
        col_desc = {
            'Hours_Studied': 'Number of hours spent studying per week',
            'Attendance': 'Percentage of classes attended',
            'Parental_Involvement': 'Level of parental involvement (Low / Medium / High)',
            'Access_to_Resources': 'Access to educational resources (Low / Medium / High)',
            'Extracurricular_Activities': 'Participation in extracurricular activities (Yes / No)',
            'Sleep_Hours': 'Average number of sleep hours per night',
            'Previous_Scores': 'Scores from previous exams',
            'Motivation_Level': 'Student motivation level (Low / Medium / High)',
            'Internet_Access': 'Access to internet (Yes / No)',
            'Tutoring_Sessions': 'Number of tutoring sessions attended per month',
            'Family_Income': 'Family income level (Low / Medium / High)',
            'Teacher_Quality': 'Quality of teachers (Low / Medium / High)',
            'School_Type': 'Type of school attended (Public / Private)',
            'Peer_Influence': 'Influence of peers on performance (Positive / Neutral / Negative)',
            'Physical_Activity': 'Average hours of physical activity per week',
            'Learning_Disabilities': 'Presence of learning disabilities (Yes / No)',
            'Parental_Education_Level': 'Highest education level of parents (High School / College / Postgraduate)',
            'Distance_from_Home': 'Distance from home to school (Near / Moderate / Far)',
            'Gender': 'Gender of the student (Male / Female)',
            'Exam_Score': '🎯 Target: Final exam score (0–100)',
        }
        for col, desc in col_desc.items():
            st.write(f"**{col}**: {desc}")

    st.markdown("##### Data Preview")
    rows = st.slider("Select number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    with st.expander("Filter data by column"):
        filt_col = st.selectbox("Column", df.columns, index=0)
        filt_val = st.text_input("Value contains (case-insensitive)")
        if filt_val:
            filtered = df[df[filt_col].astype(str).str.contains(filt_val, case=False)]
            st.dataframe(filtered)
        else:
            st.write("Enter a value to filter the table")

    st.markdown("##### Data Shape")
    st.write("Student Performance Data:", df.shape)

    st.markdown("##### Data Cleaning Summary")
    st.write(f"Original rows read: {original_row_count}")

    st.markdown("##### Missing Values")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) > 0:
        st.write(missing_values)
        st.warning(f"⚠️ {missing_values.sum()} missing values found")
    else:
        st.success("✅ No missing values found")

    st.markdown("##### Summary Statistics")
    numeric_cols_summary = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[numeric_cols_summary].describe())


elif page == "Visualization 📊":

    st.subheader("📊 Data Visualization")

    # sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.write("### Filters")
    genders = df['Gender'].unique().tolist()
    school_types = df['School_Type'].unique().tolist()
    motivations = df['Motivation_Level'].unique().tolist()

    gender_filter = st.sidebar.multiselect("Gender", genders, default=genders)
    school_filter = st.sidebar.multiselect("School Type", school_types, default=school_types)
    motivation_filter = st.sidebar.multiselect("Motivation Level", motivations, default=motivations)

    @st.cache_data
    def apply_filters(df_in, genders, schools, motivations):
        return df_in[
            (df_in['Gender'].isin(genders)) &
            (df_in['School_Type'].isin(schools)) &
            (df_in['Motivation_Level'].isin(motivations))
        ]

    df_vis = apply_filters(df, gender_filter, school_filter, motivation_filter)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Score Distribution 📊",
        "Key Factor Analysis 📋",
        "Correlation Matrix 🔥",
        "Categorical Breakdowns 📡",
        "Study & Attendance 📈",
        "Data Quality 🔍"
    ])

    with tab1:
        st.subheader("Exam Score Distribution")
        st.write("This histogram shows how exam scores are distributed across the student population.")

        fig1, ax1 = plt.subplots(figsize=(9, 5))
        sns.histplot(df_vis['Exam_Score'], bins=30, kde=True, color='#3b82f6', ax=ax1)
        ax1.set_title("Distribution of Exam Scores", fontsize=16)
        ax1.set_xlabel("Exam Score")
        ax1.set_ylabel("Number of Students")
        st.pyplot(fig1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", f"{len(df_vis):,}")
        col2.metric("Average Score", f"{df_vis['Exam_Score'].mean():.1f}")
        col3.metric("Std Deviation", f"{df_vis['Exam_Score'].std():.1f}")

        # Score band breakdown
        bins = [0, 60, 65, 70, 75, 102]
        labels = ['<60', '60–65', '65–70', '70–75', '75+']
        df_vis_copy = df_vis.copy()
        df_vis_copy['Score Band'] = pd.cut(df_vis_copy['Exam_Score'], bins=bins, labels=labels, right=False)
        band_counts = df_vis_copy['Score Band'].value_counts().sort_index()
        fig_band, ax_band = plt.subplots(figsize=(8, 4))
        colors_band = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6']
        ax_band.bar(band_counts.index.astype(str), band_counts.values, color=colors_band)
        ax_band.set_title("Students by Score Band")
        ax_band.set_xlabel("Score Range")
        ax_band.set_ylabel("Count")
        st.pyplot(fig_band)

    with tab2:
        st.subheader("Score by Key Factors")
        st.write("Compare average exam scores across different student characteristics.")

        factor = st.selectbox("Select a factor", [
            'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
            'Teacher_Quality', 'Peer_Influence', 'Family_Income',
            'School_Type', 'Internet_Access', 'Extracurricular_Activities',
            'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home'
        ])

        avg_scores = df_vis.groupby(factor)['Exam_Score'].mean().sort_values(ascending=False).reset_index()
        avg_scores.columns = [factor, 'Average Exam Score']

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        palette = sns.color_palette("Blues_r", n_colors=len(avg_scores))
        sns.barplot(data=avg_scores, x=factor, y='Average Exam Score', ax=ax2, palette=palette)
        ax2.set_title(f"Average Exam Score by {factor.replace('_', ' ')}", fontsize=15)
        ax2.set_ylabel("Average Exam Score")
        ax2.set_xlabel(factor.replace('_', ' '))
        ax2.set_ylim(avg_scores['Average Exam Score'].min() - 2, avg_scores['Average Exam Score'].max() + 2)
        st.pyplot(fig2)

        st.dataframe(avg_scores.set_index(factor))

    with tab3:
        st.subheader("Correlation Matrix")
        st.write("Heatmap of correlations between numeric features and the target exam score.")

        df_corr = df_vis.copy()
        for col in df_corr.select_dtypes(include='object').columns:
            df_corr[col] = df_corr[col].astype('category').cat.codes
        numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_corr[numeric_cols].corr()

        fig3, ax3 = plt.subplots(figsize=(11, 7))
        sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f", ax=ax3, linewidths=0.5)
        ax3.set_title("Correlation Heatmap", fontsize=15)
        st.pyplot(fig3)

        # highlight top correlations with exam score
        if 'Exam_Score' in corr.columns:
            top_corr = corr['Exam_Score'].drop('Exam_Score').sort_values(key=abs, ascending=False).head(5)
            st.markdown("**Top 5 correlations with Exam Score:**")
            st.dataframe(top_corr.rename("Correlation").to_frame())

    with tab4:
        st.subheader("Score Distribution by Category")
        st.write("Box plots showing how exam scores vary across different categorical groups.")

        cat_factor = st.selectbox("Select categorical variable", [
            'Motivation_Level', 'Parental_Involvement', 'Access_to_Resources',
            'School_Type', 'Gender', 'Peer_Influence', 'Family_Income',
            'Teacher_Quality', 'Internet_Access', 'Learning_Disabilities',
            'Extracurricular_Activities'
        ])

        fig4, ax4 = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df_vis, x=cat_factor, y='Exam_Score', ax=ax4, palette='Blues')
        ax4.set_title(f"Exam Score by {cat_factor.replace('_', ' ')}", fontsize=15)
        ax4.set_xlabel(cat_factor.replace('_', ' '))
        ax4.set_ylabel("Exam Score")
        st.pyplot(fig4)

    with tab5:
        st.subheader("Study Habits & Attendance")
        st.write("Explore how hours studied and attendance relate to exam performance.")

        fig5, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(df_vis['Hours_Studied'], df_vis['Exam_Score'], alpha=0.3, color='#3b82f6', s=10)
        m, b = np.polyfit(df_vis['Hours_Studied'], df_vis['Exam_Score'], 1)
        x_line = np.linspace(df_vis['Hours_Studied'].min(), df_vis['Hours_Studied'].max(), 100)
        axes[0].plot(x_line, m * x_line + b, color='#ef4444', linewidth=2, label=f'Trend (slope={m:.2f})')
        axes[0].set_title("Hours Studied vs Exam Score")
        axes[0].set_xlabel("Hours Studied per Week")
        axes[0].set_ylabel("Exam Score")
        axes[0].legend()

        axes[1].scatter(df_vis['Attendance'], df_vis['Exam_Score'], alpha=0.3, color='#22c55e', s=10)
        m2, b2 = np.polyfit(df_vis['Attendance'], df_vis['Exam_Score'], 1)
        x_line2 = np.linspace(df_vis['Attendance'].min(), df_vis['Attendance'].max(), 100)
        axes[1].plot(x_line2, m2 * x_line2 + b2, color='#ef4444', linewidth=2, label=f'Trend (slope={m2:.2f})')
        axes[1].set_title("Attendance % vs Exam Score")
        axes[1].set_xlabel("Attendance (%)")
        axes[1].set_ylabel("Exam Score")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig5)

        # Sleep hours analysis
        st.markdown("#### Sleep Hours vs Exam Score")
        sleep_avg = df_vis.groupby('Sleep_Hours')['Exam_Score'].mean().reset_index()
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=sleep_avg, x='Sleep_Hours', y='Exam_Score', marker='o', color='#8b5cf6', ax=ax6)
        ax6.set_title("Average Exam Score by Sleep Hours")
        ax6.set_xlabel("Sleep Hours per Night")
        ax6.set_ylabel("Average Exam Score")
        st.pyplot(fig6)

    with tab6:
        st.subheader("Data Quality Checks")

        numeric_check_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_check_cols:
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.histplot(df_vis[col].dropna(), kde=False, ax=ax, color='#3b82f6')
            ax.set_title(f"Distribution of {col.replace('_', ' ')}")
            st.pyplot(fig)

            mean = df_vis[col].mean()
            std = df_vis[col].std()
            outliers = df_vis[(df_vis[col] < mean - 3 * std) | (df_vis[col] > mean + 3 * std)]
            if not outliers.empty:
                st.write(f"⚠️ {len(outliers)} extreme values found in {col} (beyond 3σ)")
            else:
                st.success(f"✅ No outliers in {col}")

        st.markdown("---")
        st.markdown("#### Categorical Value Distributions")
        cat_cols = df_vis.select_dtypes(include='object').columns.tolist()
        col_a, col_b = st.columns(2)
        for i, col in enumerate(cat_cols):
            with (col_a if i % 2 == 0 else col_b):
                counts = df_vis[col].value_counts()
                fig_p, ax_p = plt.subplots(figsize=(5, 5))
                palette = sns.color_palette("Blues_r", n_colors=len(counts))
                ax_p.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                         colors=palette, textprops={'fontsize': 9})
                ax_p.set_title(col.replace('_', ' '), fontsize=11)
                st.pyplot(fig_p)


elif page == "Prediction 🤖":
    st.subheader("Exam Score Prediction — Linear Regression")

    test_size = 0.2

    df2 = df.dropna().copy()

    categorical_cols = [
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
                    'Tutoring_Sessions', 'Physical_Activity']

    X = df2[numeric_cols + categorical_cols]
    y = df2['Exam_Score']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    model = Pipeline(steps=[
        ('pre', preprocessor),
        ('reg', Ridge(alpha=1.0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.markdown("### Model Performance")
    st.markdown(
        "**Regression metrics for predicting the continuous Exam Score:**\n\n"
        "- **R² Score:** proportion of variance explained (1.0 is perfect).\n"
        "- **MAE:** average absolute prediction error in score points.\n"
        "- **RMSE:** root mean squared error, penalising large errors more heavily."
    )
    col1, col2, col3 = st.columns(3)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.2f} pts")
    col3.metric("RMSE", f"{rmse:.2f} pts")

    # Actual vs Predicted
    st.markdown("### Actual vs Predicted Scores")
    fig_avp, ax_avp = plt.subplots(figsize=(8, 5))
    ax_avp.scatter(y_test, y_pred, alpha=0.4, s=12, color='#3b82f6')
    lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
    ax_avp.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax_avp.set_xlabel("Actual Exam Score")
    ax_avp.set_ylabel("Predicted Exam Score")
    ax_avp.set_title("Actual vs Predicted Exam Scores")
    ax_avp.legend()
    st.pyplot(fig_avp)

    # Residual Distribution
    st.markdown("### Residual Distribution")
    residuals = y_test - y_pred
    fig_res, ax_res = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, bins=30, kde=True, color='#8b5cf6', ax=ax_res)
    ax_res.axvline(0, color='red', linestyle='--')
    ax_res.set_title("Distribution of Prediction Errors (Residuals)")
    ax_res.set_xlabel("Residual (Actual − Predicted)")
    st.pyplot(fig_res)

    # Feature Importance
    st.markdown("### Feature Importance")
    cat_names = model.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_names)
    coefs = model.named_steps['reg'].coef_

    importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    importance = importance.sort_values('Coefficient', key=abs, ascending=False).head(20)

    fig_coef, ax_coef = plt.subplots(figsize=(10, 8))
    colors = ['#ef4444' if c > 0 else '#3b82f6' for c in importance['Coefficient']]
    sns.barplot(data=importance, x='Coefficient', y='Feature', palette=colors, ax=ax_coef)
    ax_coef.set_title("Top 20 Feature Importances (Ridge Regression Coefficients)", fontsize=14)
    ax_coef.set_xlabel("Coefficient Value (positive = higher score, negative = lower score)")
    st.pyplot(fig_coef)

    st.markdown("#### Top Feature Interpretation")
    for feat, coef in importance.head(5).itertuples(index=False):
        direction = 'increases' if coef > 0 else 'decreases'
        st.write(f"- **{feat.replace('_', ' ')}**: {direction} exam score as its value rises ({coef:.3f}).")

    # Single student prediction
    st.markdown("### Predict Score for a New Student")
    with st.form(key='single_pred'):
        entries = {}
        col_a, col_b = st.columns(2)
        with col_a:
            entries['Hours_Studied'] = st.number_input("Hours Studied (per week)", value=int(df['Hours_Studied'].mean()), min_value=0, max_value=50)
            entries['Attendance'] = st.number_input("Attendance (%)", value=int(df['Attendance'].mean()), min_value=0, max_value=100)
            entries['Sleep_Hours'] = st.number_input("Sleep Hours (per night)", value=int(df['Sleep_Hours'].mean()), min_value=0, max_value=12)
            entries['Previous_Scores'] = st.number_input("Previous Scores", value=int(df['Previous_Scores'].mean()), min_value=0, max_value=100)
            entries['Tutoring_Sessions'] = st.number_input("Tutoring Sessions (per month)", value=int(df['Tutoring_Sessions'].mean()), min_value=0, max_value=20)
            entries['Physical_Activity'] = st.number_input("Physical Activity (hrs/week)", value=int(df['Physical_Activity'].mean()), min_value=0, max_value=10)
        with col_b:
            for col in categorical_cols:
                options = sorted(df[col].unique().tolist())
                entries[col] = st.selectbox(col.replace('_', ' '), options)

        submit = st.form_submit_button("Predict Exam Score")
        if submit:
            new_df = pd.DataFrame([entries])
            predicted_score = model.predict(new_df)[0]
            st.success(f"🎓 Predicted Exam Score: **{predicted_score:.1f}**")
            if predicted_score >= 75:
                st.balloons()
                st.info("🌟 This student is predicted to perform above average!")
            elif predicted_score < 63:
                st.warning("⚠️ This student may need additional support.")


elif page == "Insights and Recommendations 🧠":
    st.subheader("Insights and Recommendations")

    avg_score = df['Exam_Score'].mean()
    std_score = df['Exam_Score'].std()
    n = len(df)
    ci = 1.96 * std_score / np.sqrt(n)
    avg_study = df['Hours_Studied'].mean()
    avg_attend = df['Attendance'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Exam Score", f"{avg_score:.1f}", delta=f"±{ci:.2f} (95% CI)")
        with st.expander("Score distribution"):
            figc, axc = plt.subplots()
            sns.histplot(df['Exam_Score'], bins=30, kde=True, ax=axc, color='#3b82f6')
            axc.set_title("Exam Score Distribution")
            st.pyplot(figc)
    with col2:
        st.metric("Avg Hours Studied", f"{avg_study:.1f} hrs/wk")
        with st.expander("Hours studied distribution"):
            figh, axh = plt.subplots()
            sns.histplot(df['Hours_Studied'], bins=20, ax=axh, color='#22c55e')
            axh.set_title("Hours Studied")
            st.pyplot(figh)
    with col3:
        st.metric("Avg Attendance", f"{avg_attend:.1f}%")
        with st.expander("Attendance distribution"):
            figa, axa = plt.subplots()
            sns.histplot(df['Attendance'], bins=20, ax=axa, color='#f59e0b')
            axa.set_title("Attendance %")
            st.pyplot(figa)

    st.download_button("Download full dataset", df.to_csv(index=False), "student_performance.csv", "text/csv")

    # Score by motivation
    st.markdown("### Score by Motivation Level")
    order = ['Low', 'Medium', 'High']
    order_present = [o for o in order if o in df['Motivation_Level'].unique()]
    mot_avg = df.groupby('Motivation_Level')['Exam_Score'].mean().reindex(order_present)
    fig_mot, ax_mot = plt.subplots(figsize=(7, 4))
    palette_mot = sns.color_palette("Blues", n_colors=len(order_present))
    ax_mot.bar(mot_avg.index, mot_avg.values, color=palette_mot)
    ax_mot.set_title("Average Exam Score by Motivation Level")
    ax_mot.set_ylabel("Average Exam Score")
    ax_mot.set_ylim(mot_avg.min() - 2, mot_avg.max() + 2)
    st.pyplot(fig_mot)

    st.markdown("""
    ## 🔍 Key Insights

    1. **Hours Studied is the strongest predictor:** Each additional hour of study per week is positively associated with higher exam scores.
    2. **Attendance matters significantly:** Students with higher attendance rates consistently outperform peers with lower attendance.
    3. **Motivation drives performance:** High-motivation students score noticeably better than low-motivation peers — addressing motivation is as important as academic support.
    4. **Parental involvement has a meaningful impact:** Students with high parental involvement tend to perform better, underlining the role of the home environment.
    5. **Peer influence is real:** Students with positive peer influence outperform those with negative peer environments.
    6. **Access to resources and internet:** Students with higher access to learning materials and internet connectivity perform better, highlighting an equity gap.
    7. **Teacher quality:** Schools with higher teacher quality show improved average scores — investment in educators pays off.
    8. **Previous scores are predictive:** Prior performance is one of the strongest signals for future performance.

    ## 📋 Recommendations

    1. **Structured study programs:** Encourage and support at least 20+ hours of study per week through school-led initiatives and homework clubs.
    2. **Attendance incentives:** Implement attendance tracking with early-warning systems and incentive programmes for consistent attendance.
    3. **Motivation interventions:** Introduce mentoring, goal-setting workshops, and recognition programmes to boost student motivation.
    4. **Parental engagement:** Schools should host regular workshops and communication channels to involve parents in their child's academic journey.
    5. **Bridge the resource gap:** Provide subsidised access to learning materials, devices, and internet connectivity for students from lower-income families.
    6. **Peer learning programmes:** Foster positive peer communities through group study and collaborative projects.
    7. **Teacher development investment:** Prioritise ongoing professional development and competitive compensation to retain high-quality teachers.
    8. **Early identification of at-risk students:** Use predictive modelling (as shown in the Prediction page) to flag students likely to underperform and provide early targeted support.
    """)

    st.info("🎯 A holistic approach addressing study habits, motivation, resources, and parental involvement can improve average exam scores by several points across a cohort.")