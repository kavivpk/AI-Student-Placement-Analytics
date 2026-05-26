import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import plotly.figure_factory as ff
import plotly.graph_objects as go

# ── 1. Load Data ──────────────────────────────────
df = pd.read_csv('data/students_cleaned.csv')
print(f"✅ Data loaded: {df.shape}")

# ── 2. Features & Target ──────────────────────────
features = ['cgpa', 'attendance', 'coding_score',
            'aptitude_score', 'communication_score',
            'projects_count', 'internships']

X = df[features]
y = df['placement_encoded']

print(f"📊 Features: {features}")
print(f"📊 Target distribution:\n{y.value_counts()}")

# ── 3. Train/Test Split ───────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✅ Train size: {len(X_train)}")
print(f"✅ Test size : {len(X_test)}")

# ── 4. Scale Features ─────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train Model ────────────────────────────────
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
print("\n✅ Model trained!")

# ── 6. Evaluate ───────────────────────────────────
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy*100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Not Placed', 'Placed']))

# ── 7. Confusion Matrix Chart ─────────────────────
cm = confusion_matrix(y_test, y_pred)
fig = ff.create_annotated_heatmap(
    z=cm,
    x=['Predicted: Not Placed', 'Predicted: Placed'],
    y=['Actual: Not Placed',    'Actual: Placed'],
    colorscale='Greens'
)
fig.update_layout(title='Confusion Matrix')
fig.show()

# ── 8. Feature Importance Chart ───────────────────
importance = pd.DataFrame({
    'Feature'   : features,
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=True)

fig2 = go.Figure(go.Bar(
    x=importance['Importance'],
    y=importance['Feature'],
    orientation='h',
    marker_color='#2ecc71'
))
fig2.update_layout(title='Feature Importance')
fig2.show()

# ── 9. Save Model & Scaler ────────────────────────
with open('models/placement_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved  → models/placement_model.pkl")
print("✅ Scaler saved → models/scaler.pkl")
print("\n🎉 ML Phase Complete!")