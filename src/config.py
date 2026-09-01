"""
Configuration centralisée du projet.
Modifier ce fichier pour ajuster le pipeline sans toucher au code métier.
"""

from pathlib import Path

# ------------------------------------------------------------
# Chemins
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "donnees_sinistres.csv"
RESULTS_DIR = ROOT_DIR / "results"
INTERACTIVE_RESULTS_DIR = RESULTS_DIR / "interactive"
MODEL_PATH = RESULTS_DIR / "random_forest_model.joblib"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STREAMLIT_DEMO_URL = "https://share.streamlit.io"

# ------------------------------------------------------------
# Colonnes
# ------------------------------------------------------------

# Nom de la colonne cible dans le CSV source, et nom de la variable
# binaire créée en interne par le pipeline.
TARGET_COLUMN = "FRAUD_REPORTED"
TARGET_NAME = "SINISTRE_CIBLE"

# TARGET_MODE définit comment TARGET_COLUMN est transformée en 0/1 :
#   - "amount_threshold" : cible = 1 si TARGET_COLUMN > 0 (cas d'un montant réglé)
#   - "label"             : cible = 1 si TARGET_COLUMN == TARGET_POSITIVE_LABEL (cas d'une étiquette Y/N)
TARGET_MODE = "label"
TARGET_POSITIVE_LABEL = "Y"

DATE_COLUMNS = [
    "DATE_NAISSANCE_CLIENT",
    "DATE_EFFET",
    "DATE_SURVENANCE",
]

# Mapping des colonnes du CSV source vers les noms internes attendus
# par le pipeline. Permet de brancher un nouveau dataset sans toucher
# au code de preprocessing : il suffit d'adapter ce dictionnaire.
#
# Ici : dataset "insurance_claims" (Derrick Mwiti, GitHub), 1000 sinistres
# auto réels, cible = fraude déclarée (proxy réaliste de risque, faute de
# disposer de contrats sans sinistre dans ce jeu de données).
RENAME_MAP = {
    "AGE": "AGE_ASSURE",
    "MONTHS_AS_CUSTOMER": "DUREE_MOIS_CONTRAT",
    "POLICY_ANNUAL_PREMIUM": "PRIME",
    "UMBRELLA_LIMIT": "CAPITAL_ASSURE",
    "TOTAL_CLAIM_AMOUNT": "SINISTRE_REGLE",  # conservée à titre informatif
}

# Variables explicatives candidates.
# Seules celles réellement présentes dans le CSV (après renommage) seront conservées.
CANDIDATE_FEATURES = [
    "AGE_ASSURE",
    "CAPITAL_ASSURE",
    "PRIME",
    "DUREE_MOIS_CONTRAT",
]

# ------------------------------------------------------------
# Split train/test
# ------------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# ------------------------------------------------------------
# Recherche d'hyperparamètres (RandomizedSearchCV)
# ------------------------------------------------------------

N_ITER_SEARCH = 30

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
}

# ------------------------------------------------------------
# Bornes raisonnables pour la validation des données
# ------------------------------------------------------------

AGE_MIN = 0
AGE_MAX = 120
