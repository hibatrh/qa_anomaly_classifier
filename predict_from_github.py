import sys
import os
import joblib

# ---------------------------
# Charger le modèle
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models/priority_classifier.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/vectorizer.pkl"))

# ---------------------------
# Script principal
# ---------------------------
if __name__ == "__main__":

    # Vérifier qu'on a bien deux arguments (titre et corps)
    if len(sys.argv) < 3:
        print("Usage: python predict_from_github.py <title> <body>")
        sys.exit(1)

    # Récupérer le titre et le corps de l'issue
    title = sys.argv[1]
    body = sys.argv[2]

    # ---------------------------
    # Prediction
    # ---------------------------
    text = f"{title} {body}"
    X = vectorizer.transform([text])
    priority = model.predict(X)[0]
    print(f"Prediction: {priority}")

    # ---------------------------
    # Sortie pour GitHub Actions
    # ---------------------------
    if "GITHUB_OUTPUT" in os.environ:
        # Écrire la sortie dans le fichier GITHUB_OUTPUT pour le workflow
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"priority={priority}\n")
    else:
        # Pour tests locaux ou anciens GitHub Actions
        print(f"::set-output name=priority::{priority}")
