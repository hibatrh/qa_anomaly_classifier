import joblib
import sys

# ---------------------------
# Charger le modèle et le vectorizer
# ---------------------------
model = joblib.load("models/priority_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ---------------------------
# Fonction de prédiction
# ---------------------------
def predict_priority(title, description):
    """Prédit la priorité d’une nouvelle issue"""
    
    # Fusionner le titre et la description
    text = f"{title} {description}"
    
    # Transformer le texte en vecteur TF-IDF
    X = vectorizer.transform([text])
    
    # Prédire la priorité
    priority = model.predict(X)[0]
    
    # Prédire les probabilités pour chaque classe
    probabilities = model.predict_proba(X)[0]
    
    # Retourner le résultat sous forme de dictionnaire
    return {
        "priority": priority,
        "confidence": max(probabilities) * 100,  # Pourcentage de confiance
        "probabilities": dict(zip(model.classes_, probabilities))  # Probabilités détaillées
    }

# ---------------------------
# Exemple d’utilisation
# ---------------------------
if __name__ == "__main__":

    # --- Test 1 : Bug critique ---
    result1 = predict_priority(
        "Critical memory leak causing crash",
        "Application crashes after 1 hour of usage. Memory grows continuously."
    )
    print("Prediction 1:")
    print(f" Priorite: {result1['priority']}")
    print(f" Confiance: {result1['confidence']:.1f}%")
    print(f" Details: {result1['probabilities']}\n")

    # --- Test 2 : Feature request ---
    result2 = predict_priority(
        "Add dark mode support",
        "Users would like a dark theme option in settings"
    )
    print("Prediction 2:")
    print(f" Priorite: {result2['priority']}")
    print(f" Confiance: {result2['confidence']:.1f}%")
    print(f" Details: {result2['probabilities']}")
