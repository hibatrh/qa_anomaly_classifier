import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Charger le CSV
df = pd.read_csv("data/issues.csv")

# Créer la colonne 'text' en combinant 'title' et 'body'
df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

# Générer les embeddings
print("Génération des embeddings...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
X = bert_model.encode(df["text"].tolist())

# Labels
y = df["priority"]

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le classifieur
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# Éventuellement, sauvegarder le modèle
import joblib
joblib.dump(classifier, "models/bert_priority_classifier.pkl")
print("Modèle BERT + Random Forest entraîné et sauvegardé !")
