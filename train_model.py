import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------
# Charger les données
# ---------------------------
print("Chargement des données...")
df = pd.read_csv("data/issues.csv")

# ---------------------------
# Nettoyer les données
# ---------------------------
# On supprime les lignes où 'title' ou 'body' est vide
df = df.dropna(subset=["title", "body"])

# Fusionner le titre et le corps dans une seule colonne 'text'
df["text"] = df["title"] + " " + df["body"]

print(f"Total issues: {len(df)}")
print(f"Distribution:\n{df['priority'].value_counts()}")

# ---------------------------
# Vectorisation TF-IDF
# ---------------------------
print("\nVectorisation TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=500,         # Limite à 500 mots les plus significatifs
    ngram_range=(1, 2),       # Unigrammes et bigrammes
    stop_words="english"      # Supprime les mots vides anglais
)

# Transformer le texte en matrice TF-IDF
X = vectorizer.fit_transform(df["text"])
y = df["priority"]

# ---------------------------
# Split train/test
# ---------------------------
# 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------
# Entrainement du modèle Random Forest
# ---------------------------
print("\nEntrainement Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,  # Nombre d'arbres
    max_depth=10,      # Profondeur maximale
    random_state=42,
    n_jobs=-1          # Utiliser tous les coeurs CPU disponibles
)

model.fit(X_train, y_train)

# ---------------------------
# Prédictions sur le test set
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Evaluation
# ---------------------------
print("\nRésultats:")
print(classification_report(y_test, y_pred))

# ---------------------------
# Matrice de confusion
# ---------------------------
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.title("Matrice de confusion - Prédiction de priorité")
plt.ylabel("Vraie priorité")
plt.xlabel("Priorité prédite")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=300)
print("Matrice sauvegardée: models/confusion_matrix.png")

# ---------------------------
# Sauvegarder le modèle et le vectorizer
# ---------------------------
joblib.dump(model, "models/priority_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("Modèle sauvegardé: models/priority_classifier.pkl")
