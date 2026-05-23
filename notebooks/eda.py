import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('data/students_cleaned.csv')

# ── 1. Placement Count ──────────────────────────────
fig1 = px.pie(
    df,
    names='placement_status',
    title='Overall Placement Status',
    color_discrete_sequence=['#2ecc71','#e74c3c']
)
fig1.show()

# ── 2. Department-wise Placement ────────────────────
dept = df.groupby(['department','placement_status']).size().reset_index(name='count')
fig2 = px.bar(
    dept,
    x='department',
    y='count',
    color='placement_status',
    title='Department-wise Placement',
    barmode='group',
    color_discrete_sequence=['#2ecc71','#e74c3c']
)
fig2.show()

# ── 3. CGPA vs Placement ────────────────────────────
fig3 = px.box(
    df,
    x='placement_status',
    y='cgpa',
    color='placement_status',
    title='CGPA vs Placement Status',
    color_discrete_sequence=['#2ecc71','#e74c3c']
)
fig3.show()

# ── 4. Coding Score Distribution ────────────────────
fig4 = px.histogram(
    df,
    x='coding_score',
    color='placement_status',
    title='Coding Score Distribution',
    nbins=20,
    color_discrete_sequence=['#2ecc71','#e74c3c']
)
fig4.show()

# ── 5. Correlation Heatmap ──────────────────────────
cols = ['cgpa','attendance','coding_score',
        'aptitude_score','communication_score',
        'projects_count','internships','placement_encoded']
corr = df[cols].corr().round(2)

fig5 = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.columns.tolist(),
    colorscale='RdYlGn',
    text=corr.values,
    texttemplate='%{text}'
))
fig5.update_layout(title='Feature Correlation Heatmap')
fig5.show()

# ── Basic Stats ──────────────────────────────────────
print("=== Basic Statistics ===")
print(f"Total Students  : {len(df)}")
print(f"Placed          : {(df['placement_status']=='Placed').sum()}")
print(f"Not Placed      : {(df['placement_status']=='Not Placed').sum()}")
print(f"Placement %     : {(df['placement_status']=='Placed').mean()*100:.1f}%")
print(f"Average CGPA    : {df['cgpa'].mean():.2f}")
print(f"Avg Coding Score: {df['coding_score'].mean():.1f}")