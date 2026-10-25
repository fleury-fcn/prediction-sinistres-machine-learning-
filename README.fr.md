# Détection de fraude sur sinistres automobiles : un benchmark sans fuite et statistiquement honnête

[![CI](https://github.com/fleury-fcn/prediction-sinistres-machine-learning-/actions/workflows/ci.yml/badge.svg)](https://github.com/fleury-fcn/prediction-sinistres-machine-learning-/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![Données publiques](https://img.shields.io/badge/donn%C3%A9es-benchmark%20public-lightgrey)

[🇬🇧 English version](README.md) · **🇫🇷 Français**

**En trois lignes.** Sur le jeu public de 1 000 sinistres, neuf modèles optimisés avec le même
budget Optuna obtiennent des PR-AUC à quelques points les unes des autres. Le meilleur (CatBoost,
choisi par validation croisée) atteint **une ROC-AUC de 0,84 et une PR-AUC de 0,55 sur le jeu de
test jamais utilisé**, réduit le coût moyen par sinistre de **32 %** par rapport à une enquête
systématique (selon les hypothèses de coût affichées), et **n'est pas significativement
meilleur** qu'une forêt aléatoire, un gradient boosting ou une régression logistique elastic net
bien réglés.

## Résultats principaux (données réelles, jeu de test de 200 sinistres dont 49 fraudes)

| Modèle | PR-AUC CV (moyenne ± é.-t.) | PR-AUC test [IC 95 %] | ROC-AUC test [IC 95 %] | F1 | Rappel | Précision | MCC | Brier | Coût / sinistre (USD) |
|---|---|---|---|---|---|---|---|---|---|
| **CatBoost** (référence, exporté) | 0,707 ± 0,054 | 0,553 [0,463 ; 0,708] | 0,839 [0,769 ; 0,902] | 0,730 | 0,857 | 0,636 | 0,639 | 0,124 | 680 |
| Stacking (top 3 + méta logistique) | 0,704 ± 0,062 | 0,588 [0,482 ; 0,721] | 0,827 [0,747 ; 0,895] | 0,730 | 0,857 | 0,636 | 0,639 | 0,126 | 680 |
| Forêt aléatoire | 0,701 ± 0,061 | 0,600 [0,489 ; 0,736] | 0,827 [0,748 ; 0,897] | 0,718 | 0,857 | 0,618 | 0,622 | 0,123 | 690 |
| Gradient boosting (sklearn) | 0,695 ± 0,057 | 0,592 [0,483 ; 0,726] | 0,837 [0,766 ; 0,899] | 0,730 | 0,857 | 0,636 | 0,639 | 0,120 | 680 |
| Régression logistique (elastic net) | 0,693 ± 0,054 | 0,524 [0,434 ; 0,669] | 0,818 [0,743 ; 0,884] | 0,609 | 0,857 | 0,472 | 0,472 | 0,152 | 795 |
| XGBoost | 0,683 ± 0,059 | 0,512 [0,432 ; 0,657] | 0,832 [0,763 ; 0,894] | 0,726 | 0,837 | 0,641 | 0,631 | 0,143 | 720 |
| MLP + embeddings (PyTorch) | 0,668 ± 0,062 | 0,537 [0,447 ; 0,682] | 0,840 [0,777 ; 0,896] | 0,621 | 0,837 | 0,494 | 0,488 | 0,146 | 815 |
| LightGBM | 0,659 ± 0,058 | 0,520 [0,439 ; 0,665] | 0,830 [0,753 ; 0,897] | 0,707 | 0,837 | 0,612 | 0,606 | 0,140 | 735 |
| Régression logistique (sans pénalité) | 0,629 ± 0,065 | 0,499 [0,402 ; 0,646] | 0,771 [0,686 ; 0,844] | 0,503 | 0,735 | 0,383 | 0,302 | 0,175 | 1 120 |

Les lignes sont triées par PR-AUC **en validation croisée** (CV répétée 5 × 5 sur les 800
sinistres d'entraînement), le critère qui a servi à choisir le modèle de référence ; le test n'a
été utilisé qu'une fois, à la fin. F1, rappel, précision, MCC et coût utilisent le seuil
optimal en coût de chaque modèle, choisi sur ses prédictions hors-pli (OOF) d'entraînement.
Les crochets sont des intervalles bootstrap stratifiés à 95 % (2 000 rééchantillonnages).
Règles simples sur le même test : *ne rien enquêter* = 2 450 USD/sinistre, *tout enquêter* =
1 000 USD/sinistre. La prévalence de la fraude (PR-AUC d'un modèle aléatoire) vaut 0,245.

![Courbes ROC et PR](results/figures/real_roc_pr_curves.png)

![Métriques de test avec intervalles bootstrap](results/figures/real_metrics_ci.png)

![|SHAP| moyen par variable](results/figures/real_shap_by_feature.png)

![Coût attendu selon le seuil](results/figures/real_cost_vs_threshold.png)

### Ce que disent les chiffres

* **Gros gain par rapport à l'ancienne version, petits écarts entre modèles.** L'ancien
  pipeline à 4 variables obtenait une ROC-AUC ≈ 0,50 (hasard). Utiliser toutes les colonnes
  pertinentes amène tous les modèles optimisés à une ROC-AUC de 0,82 à 0,84 sur le test. Entre
  modèles, les écarts sont plus petits que l'incertitude : le t-test corrigé de Nadeau–Bengio
  sur les 25 plis de CV et le bootstrap apparié sur le test ne montrent aucune différence
  significative avec CatBoost après correction de Holm (plus petite p-value ajustée = 0,22).
  Avant correction, seuls LightGBM (p = 0,03) et la régression logistique sans pénalité
  (p = 0,04) paraissent moins bons en CV.
* **CV contre test.** La PR-AUC de test (0,50 à 0,60) est inférieure à la PR-AUC de CV (0,63 à
  0,71). Avec 49 fraudes dans le test, l'intervalle bootstrap d'une PR-AUC couvre ±0,1 : l'écart
  est compatible avec le bruit d'échantillonnage, et les chiffres de CV (200 fraudes × 5
  répétitions) sont l'estimation la plus stable.
* **Qualité des décisions.** Aux seuils retenus, les ensembles d'arbres détectent 86 % des
  fraudes avec 64 % de précision. Les tests de McNemar détectent de vraies différences d'erreurs
  à ces points de fonctionnement (CatBoost juste / elastic net faux sur 26 sinistres contre 3
  dans l'autre sens, p < 0,001), surtout parce que les modèles linéaires et le MLP signalent
  davantage de sinistres honnêtes à leur propre seuil optimal.
* **Le deep learning n'aide pas ici.** Le MLP avec embeddings catégoriels est compétitif en
  ROC-AUC de test (0,84) mais a l'avant-dernière PR-AUC en CV et le score de Brier en CV le plus
  élevé parmi les modèles optimisés. Avec 640 lignes d'entraînement par pli, c'est attendu ; il
  est gardé comme point de comparaison honnête, pas comme modèle recommandé.
* **Deux variables portent le signal** (voir l'interprétation). Retirer `insured_hobbies` fait
  passer la PR-AUC de CV de 0,71 à 0,50 ; retirer `incident_severity` la fait tomber à 0,54.

### Explorer les résultats de façon interactive

Une application Streamlit bilingue (FR/EN) présente toute l'étude : vue d'ensemble, données,
comparaison des modèles avec intervalles de confiance et tests statistiques, **simulateur de
coût métier** (modifiez le coût d'une enquête, celui d'une fraude manquée et le seuil),
interprétation SHAP, bloc données synthétiques, et **évaluation en direct d'un sinistre** avec
son explication locale.

```bash
make app        # ou : streamlit run app/streamlit_app.py
```

L'application ne lit que des fichiers versionnés et a ses propres dépendances légères
([`app/requirements.txt`](app/requirements.txt)) : elle peut être déployée gratuitement sur
[Streamlit Community Cloud](https://share.streamlit.io) avec `app/streamlit_app.py` comme point
d'entrée. Choisissez **Python 3.12 dans « Advanced settings » avant de cliquer sur Deploy** : la
version par défaut (3.14) n'a pas de paquets précompilés pour les versions épinglées, et la
version de Python d'une application ne peut plus être changée après le déploiement (il faut la
supprimer puis la redéployer).

---

## Sommaire

1. [Problème et données](#problème-et-données)
2. [Méthodologie](#méthodologie)
3. [Résultats détaillés](#résultats-détaillés)
4. [Données synthétiques : montée en charge et robustesse uniquement](#données-synthétiques--montée-en-charge-et-robustesse-uniquement)
5. [Interprétation](#interprétation)
6. [Limites et remarques honnêtes](#limites-et-remarques-honnêtes)
7. [Reproduire](#reproduire)
8. [Structure du dépôt](#structure-du-dépôt)
9. [Servir le modèle](#servir-le-modèle)
10. [Perspectives](#perspectives)

## Problème et données

La tâche consiste à évaluer le risque de fraude d'un sinistre automobile **déjà déclaré**, pour
aider un assureur à décider quels sinistres envoyer en enquête. Le jeu public de Derrick Mwiti
(<https://github.com/mwitiderrick/insurancedata>) compte 1 000 sinistres, 39 colonnes et 247
fraudes (24,7 %) : contrat (ancienneté, État, franchise, prime, plafond umbrella, limite
combinée), assuré (âge, sexe, études, profession, loisirs, lien, plus- et moins-values),
incident (date, type, type de collision, gravité, autorités contactées, État, ville, heure,
véhicules, dommages matériels, blessés, témoins, rapport de police), montants réclamés
(corporel, matériel, véhicule, total) et véhicule (marque, modèle, année).

Constats de qualité établis dans [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) :

* `?` apparaît dans `collision_type` (178), `property_damage` (360) et
  `police_report_available` (343). Il est conservé comme niveau explicite `UNKNOWN` : pour
  `collision_type`, il correspond exactement aux vols et aux véhicules stationnés (valeur
  manquante structurelle, pas aléatoire).
* `authorities_contacted` contient le niveau littéral `"None"` (91 sinistres) que pandas
  transforme en NaN par défaut ; le chargement lit le CSV avec `keep_default_na=False`.
* Un `umbrella_limit` négatif est traité comme une erreur de saisie (NaN, imputé dans le
  pipeline).
* Aucun doublon ; les incidents vont du 1er janvier au 1er mars 2015.

![Taux de fraude par catégorie](results/figures/eda_fraud_rate_by_category.png)

### Audit de fuite de `total_claim_amount`

* `total_claim_amount` vaut `injury_claim + property_claim + vehicle_claim` sur **chaque**
  ligne (écart absolu maximal 0). Il est donc redondant et supprimé ; son information reste
  présente via les composantes et la variable construite `log_total_claim`.
* Ce sont des montants **réclamés** au moment de la déclaration (les sinistres frauduleux ne
  sont pas enregistrés avec un paiement nul) : ce n'est pas une variable post-enquête. Leur
  ROC-AUC univariée est de 0,56 à 0,60, loin de la séparation quasi parfaite que donnerait une
  variable fuitant la cible.
* Ablation (CV 5 × 5, CatBoost optimisé) : toutes les variables 0,707 de PR-AUC ; en rajoutant le
  total redondant 0,705 ; **en retirant toutes les variables issues des montants 0,716**. Les
  montants n'apportent rien : ils ne peuvent pas être un canal de fuite. Ils sont gardés par
  souci d'exhaustivité.

### Construction de variables

Toutes les variables construites sont calculées ligne par ligne dans la première étape de
chaque pipeline ([`src/features.py`](src/features.py)) ; les identifiants (`policy_number`,
`insured_zip`, `incident_location`) ne sont jamais des entrées du modèle.

| Variable | Pourquoi |
|---|---|
| `policy_tenure_days` | date d'incident − date de souscription ; les contrats récents sont un signal classique de fraude et les dates brutes sont inutilisables |
| `incident_weekday` | effet du jour de la semaine ; mois et année n'ont pas de sens sur deux mois |
| `csl_per_person`, `csl_per_accident` | `policy_csl = "250/500"` cache deux limites numériques dans une chaîne |
| `vehicle_age` | plus interprétable que l'année du modèle |
| `log_total_claim`, `injury_share`, `property_share` | montants asymétriques et composition du sinistre (part véhicule omise : complément exact) |
| `claim_to_premium` | taille du sinistre rapportée à la prime payée |
| `has_umbrella` | 80 % des contrats n'ont pas de garantie umbrella ; aide les modèles linéaires à séparer « aucune » de « montant » |

## Méthodologie

```text
                       data/insurance_claims.csv (1 000 sinistres)
                                     │  nettoyage déterministe (aucune statistique apprise)
                                     ▼
              split stratifié ── test 200 (verrouillé jusqu'à l'étape evaluate)
                                     │
                              entraînement 800
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     ▼                               ▼                               ▼
 Optuna, 40 essais par modèle    CV stratifiée répétée 5×5       générateurs ajustés sur
 mêmes 5 plis, objectif PR-AUC   (params optimisés) → scores     ces 800 lignes seulement
 stratégie de déséquilibre       par pli + probabilités          (étape synthétique)
 incluse dans la recherche       hors-pli (OOF)
     │                               ├─ seuil optimal en coût par modèle (OOF)
     │                               ├─ modèle de référence = meilleure PR-AUC moyenne en CV
     │                               └─ t-test corrigé entre modèles (plis de CV)
     ▼
 réentraînement sur les 800 ──► évaluation unique sur les 200 sinistres de test
                                IC bootstrap · bootstrap apparié · McNemar · Holm
                                calibration · coûts · SHAP · importance par permutation
```

Toute étape qui apprend des données vit dans un `Pipeline` scikit-learn / imbalanced-learn
([`src/pipelines.py`](src/pipelines.py)) : imputation par la médiane, standardisation, encodage
one-hot (les niveaux présents moins de 10 fois dans le pli d'entraînement sont regroupés), SMOTE
(appliqué uniquement à l'entraînement), encodage catégoriel natif de CatBoost, et imputation,
mise à l'échelle et dictionnaires de catégories propres au MLP.
[`tests/test_leakage.py`](tests/test_leakage.py) vérifie que les statistiques apprises sont
celles du pli d'entraînement, que les catégories inconnues du test sont gérées, que SMOTE ne
s'applique jamais à la prédiction, et que l'étape d'entraînement ne lit jamais les étiquettes de
test.

**Modèles** ([`src/models.py`](src/models.py)) : régression logistique sans pénalité (baseline,
sans hyperparamètre) et avec elastic net ; forêt aléatoire ; gradient boosting de
scikit-learn ; XGBoost ; LightGBM ; CatBoost ; un MLP PyTorch avec embeddings, batch norm,
dropout, AdamW et arrêt précoce sur un sous-ensemble de validation interne
([`src/nn.py`](src/nn.py)) ; et un stacking des trois meilleurs modèles (en CV) avec un
méta-modèle logistique.

**Optimisation** : Optuna TPE (graine fixée), 40 essais par modèle optimisable, les mêmes cinq
plis pour tous, objectif = PR-AUC moyenne, élagage médian, configuration par défaut évaluée en
premier. La stratégie de déséquilibre (aucune / poids de classes / SMOTE) fait partie des
hyperparamètres recherchés quand elle s'applique.

**Coûts métier** (hypothèses, paramétrables dans [`configs/default.yaml`](configs/default.yaml)
ou avec `--cost-fn` / `--cost-investigation`) :

| Issue | Coût (USD) |
|---|---|
| sinistre signalé (vrai ou faux positif) : enquête | 1 000 |
| fraude non signalée (faux négatif) : payée | 10 000 |
| sinistre honnête non signalé | 0 |

Avec des probabilités bien calibrées, la règle optimale serait « signaler si P(fraude) > 1 000 /
10 000 = 0,10 » ; le pipeline choisit plutôt, pour chaque modèle, le seuil qui minimise le coût
sur ses prédictions OOF d'entraînement (CatBoost : 0,15).

## Résultats détaillés

Tous les fichiers sont dans [`results/metrics`](results/metrics) et proviennent de `make all`.

### Validation croisée

![PR-AUC par pli de CV](results/figures/real_cv_pr_auc_folds.png)

La dispersion d'un pli à l'autre (écart-type ≈ 0,055) est de l'ordre de l'écart entre le
premier et le septième modèle : c'est pourquoi le t-test corrigé ne trouve aucune différence
significative.

### Stratégies de gestion du déséquilibre

Mêmes hyperparamètres optimisés, seule la stratégie change (PR-AUC en CV 5 × 5 sur
l'entraînement) :

| Modèle | aucune | poids de classes | SMOTE | choix d'Optuna |
|---|---|---|---|---|
| Régression logistique (elastic net) | 0,687 | **0,693** | 0,682 | poids |
| Forêt aléatoire | 0,685 | 0,682 | **0,701** | SMOTE |
| XGBoost | 0,659 | **0,683** | 0,666 | poids |
| LightGBM | 0,499 | 0,508 | **0,659** | SMOTE |

Pour trois modèles, les stratégies sont à moins d'un écart-type les unes des autres (≈ 0,05).
L'écart de LightGBM n'est pas un pur effet du déséquilibre : son `min_child_samples = 48`
optimisé dépasse les 26 à 32 sinistres « chess » d'un pli d'entraînement, donc sans les lignes
minoritaires ajoutées par SMOTE les arbres ne peuvent pas isoler ce niveau ; la stratégie
interagit avec les autres hyperparamètres. Pour la régression logistique elastic net, la
repondération et SMOTE dégradent la calibration (Brier en CV 0,114 → 0,125 / 0,129) sans gain
net de classement ; pour les arbres, le Brier bouge à peine
([tableau complet](results/metrics/real_imbalance_strategies.csv)).

### Calibration

![Diagramme de fiabilité](results/figures/real_calibration.png)

| CatBoost sur le test | Brier |
|---|---|
| non calibré | 0,124 |
| + calibration de Platt (5 plis, entraînement seul) | 0,124 |
| + régression isotonique (5 plis, entraînement seul) | 0,122 |

Les ensembles d'arbres sont déjà raisonnablement calibrés ; la recalibration n'apporte pas de
gain significatif sur 200 sinistres de test et n'est pas utilisée dans le modèle exporté.

### Décisions au seuil optimal en coût

![Matrices de confusion](results/figures/real_confusion_matrices.png)

Sensibilité de la politique CatBoost au ratio de coûts (seuil réoptimisé sur l'OOF
d'entraînement pour chaque ratio, coût mesuré sur le test) :

| Fraude manquée / enquête | 2 | 3 | 5 | 7,5 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| seuil | 0,46 | 0,15 | 0,15 | 0,15 | 0,15 | 0,15 | 0,15 | 0,01 | 0,01 |
| économie vs meilleure règle simple | 12 % | 41 % | 50 % | 41 % | 32 % | 15 % | −3 % | 0 % | 0 % |

Le modèle n'a d'intérêt que pour des ratios de coûts intermédiaires : quand une fraude manquée
coûte 20 fois une enquête ou plus, tout enquêter fait aussi bien ou mieux.

![Sensibilité aux coûts](results/figures/real_cost_sensitivity.png)

### Courbes d'apprentissage et entraînement du réseau de neurones

![Courbes d'apprentissage](results/figures/real_learning_curves.png)

CatBoost plafonne dès environ 300 sinistres d'entraînement, alors que la régression logistique
elastic net progresse encore à 640 : davantage de données réelles aiderait surtout les modèles
simples.

![Courbes de perte du MLP](results/figures/real_mlp_loss_curves.png)

Le MLP final (une couche cachée de 32 neurones, dropout 0,5, choisis par Optuna) atteint sa
meilleure perte de validation après 10 époques puis surapprend ; l'arrêt précoce restaure
cette époque.

## Données synthétiques : montée en charge et robustesse uniquement

> ⚠️ **Tout ce qui figure dans cette section est SYNTHÉTIQUE.** Aucun chiffre ici ne mesure
> une performance réelle de détection de fraude. Les fichiers et figures synthétiques portent
> le préfixe `SYNTHETIC` / `synthetic_`.

Trois générateurs ont été ajustés **uniquement sur les 800 sinistres réels d'entraînement**,
pour que les 200 sinistres réels de test restent inédits et servent au protocole « entraîner
sur synthétique, tester sur réel » (TSTR) :

* une **copule gaussienne conditionnelle à la classe** écrite pour ce projet
  ([`src/synthetic.py`](src/synthetic.py)) : une copule par classe (taux de fraude reproduit
  exactement), marginales empiriques, niveaux catégoriels associés à des intervalles de
  fréquence ;
* **SDV GaussianCopula** et **SDV CTGAN** (300 époques).

La même correction de règle métier (un type de collision existe si et seulement si l'incident
est une collision) est appliquée à la sortie de chaque générateur ; le taux brut de violation
est rapporté.

| Contrôle qualité (20 000 lignes par générateur vs les 800 sinistres réels d'entraînement) | Copule maison | SDV GaussianCopula | SDV CTGAN |
|---|---|---|---|
| taux de fraude (réel : 0,248) | 0,248 | 0,243 | 0,329 |
| distance KS moyenne, marginales numériques (0 = identiques) | **0,010** | 0,190 | 0,176 |
| distance de variation totale moyenne, marginales catégorielles | **0,008** | 0,011 | 0,056 |
| écart absolu moyen des corrélations de Spearman | **0,014** | 0,018 | 0,065 |
| écart absolu moyen du V de Cramér (catégorielles + cible) | **0,085** | 0,100 | 0,107 |
| violations brutes de la règle métier | 21 % | 32 % | 31 % |
| ROC-AUC de détection réel vs synthétique (0,5 = indiscernable) | **0,83** | 1,00 | 1,00 |
| copies exactes de lignes réelles | 0 | 0 | 0 |
| distance à l'enregistrement réel le plus proche, ratio vs réel tenu à l'écart | 1,01 | 1,05 | 1,06 |
| TSTR : LightGBM sur 20 000 lignes synthétiques → PR-AUC sur le test réel | **0,537** | 0,423 | 0,232 |
| TSTR : idem, ROC-AUC | **0,831** | 0,716 | 0,436 |
| Référence TRTR : même LightGBM sur les 800 lignes réelles → test réel | 0,520 / 0,830 | | |

* La copule maison est le meilleur générateur sur tous les contrôles ; elle a été retenue pour
  le grand jeu sur **l'AUC de détection, un critère calculé uniquement contre l'entraînement**
  (le test réel ne joue aucun rôle dans ce choix). Avec 800 lignes, CTGAN n'apprend pas la loi
  jointe (il décale même le taux de fraude à 33 %), et les marginales paramétriques de SDV
  ajustent mal les colonnes numériques très discrètes.
* Même le meilleur générateur reste **détectable** (AUC 0,83) : reproduire les marginales et
  les corrélations deux à deux ne reproduit pas la structure d'ordre supérieur, comme le
  mélange de petits sinistres de vol et de gros sinistres de collision.
* **Confidentialité** : aucune ligne réelle n'est copiée, et les lignes synthétiques sont aussi
  éloignées des sinistres d'entraînement que le sont de vrais nouveaux sinistres (ratio de
  distance 1,01 ; 4 % des lignes synthétiques sont sous le 5e percentile des distances du réel
  tenu à l'écart, proche des 5 % attendus par hasard).
* **TSTR ≈ TRTR** : un modèle entraîné sur 20 000 sinistres synthétiques obtient 0,537 de PR-AUC
  sur le test réel, contre 0,520 pour le même modèle entraîné sur les 800 sinistres réels, un
  écart largement dans l'incertitude de ±0,1 du test. Le générateur préserve le signal ; il
  n'en ajoute pas.

![Qualité des données synthétiques](results/figures/synthetic_quality.png)

![Marginales réelles vs synthétiques](results/figures/synthetic_marginals.png)

![Corrélations réelles vs synthétiques](results/figures/synthetic_correlations.png)

### Grand jeu synthétique et benchmark de montée en charge

`make synth` écrit `data/synthetic/SYNTHETIC_claims_gaussian_copula_custom_500000.parquet`
(500 000 lignes, même schéma à 39 colonnes, taux de fraude 24,75 %, 23 Mo, ignoré par git) ainsi
qu'un échantillon versionné de 1 000 lignes. Chaque modèle est réentraîné avec ses
hyperparamètres réels sur 1k → 500k lignes synthétiques, puis évalué sur un jeu **synthétique**
tenu à l'écart de 50 000 lignes et sur les 200 sinistres **réels** de test :

| Modèle (hyperparamètres des données réelles) | Temps d'entraînement à 500k | PR-AUC, test synthétique (1k → 500k) | PR-AUC, test réel / TSTR (1k → 500k) |
|---|---|---|---|
| CatBoost | 255 s | 0,822 → 0,928 | 0,533 → 0,495 |
| MLP + embeddings | 37 s | 0,782 → 0,904 | 0,485 → 0,466 |
| LightGBM | 113 s | 0,803 → 0,862 | 0,549 → 0,524 |
| XGBoost | 39 s | 0,786 → 0,837 | 0,503 → 0,527 |
| Régression logistique (elastic net) | 477 s | 0,764 → 0,835 | 0,473 → 0,508 |
| Forêt aléatoire (plafonnée à 100k) | 896 s à 100k | 0,742 → 0,861 | 0,610 → 0,566 |
| Gradient boosting (plafonné à 100k) | 164 s à 100k | 0,772 → 0,800 | 0,590 → 0,533 |

![Benchmark de montée en charge](results/figures/synthetic_scale.png)

* **Côté ingénierie** : XGBoost et le MLP s'entraînent sur 500 000 sinistres en moins de 40 s
  sur un processeur de portable, LightGBM en 2 min, CatBoost en 4 min ; le solveur saga de la
  régression elastic net est le modèle « scalable » le plus lent (8 min). La forêt aléatoire
  de scikit-learn (800 arbres profonds + SMOTE) et le gradient boosting ont été plafonnés à
  100 000 lignes car ils passent mal à l'échelle.
* **Côté statistique** : sur le test synthétique, tous les modèles progressent avec la taille,
  et les scores (0,80 à 0,93) dépassent de loin ce qui est atteignable sur de vrais sinistres.
  Sur les sinistres **réels**, davantage de données synthétiques **n'aide pas** : les
  variations entre 1k et 500k lignes vont de −0,04 à +0,035, toutes dans l'incertitude de ±0,1
  du test, et les modèles les plus flexibles (CatBoost, MLP) baissent légèrement en apprenant
  les régularités propres au générateur. La tâche synthétique est plus facile que la vraie :
  c'est précisément pourquoi un score synthétique ne doit jamais être présenté comme une
  performance.
* Écarts par rapport au protocole réel, nécessaires pour garder des temps raisonnables :
  au-delà de 10 000 lignes, le MLP utilise des batchs de 1 024 et au plus 30 époques
  (patience 4) ; aucune réoptimisation n'a été faite sur les données synthétiques.

## Interprétation

![SHAP beeswarm](results/figures/real_shap_beeswarm.png)

Valeurs SHAP du modèle CatBoost de référence sur les sinistres de test (échelle log-odds),
confirmées par l'importance par permutation sur les mêmes sinistres (baisse de PR-AUC quand une
colonne est mélangée : `incident_severity` 0,23, `insured_hobbies` 0,14, toute autre colonne
< 0,01) :

* **`incident_severity`** est le facteur dominant : les sinistres *Major Damage* comptent
  ~60 % de fraude contre 7 à 13 % pour les autres gravités, et le modèle augmente leurs log-odds
  d'environ +2. Une lecture métier plausible : c'est sur les gros dommages que gonfler un
  sinistre rapporte, mais les données ne permettent pas de distinguer une exagération d'un
  accident réellement grave.
* **`insured_hobbies`** égal à chess ou cross-fit (83 % et 74 % de fraude) est le deuxième
  facteur. Aucun mécanisme causal crédible n'explique cela : c'est presque certainement un
  artefact de la fabrication de ce jeu public. Un modèle déployé sur un vrai portefeuille ne le
  retrouverait sans doute pas, et un modèle de production devrait probablement exclure ce type
  de variable pour des raisons d'équité et de robustesse.
* Tout le reste (part corporelle du sinistre, témoins, âge du véhicule, profession, État) pèse
  un ordre de grandeur de moins.

![Explications locales](results/figures/real_shap_local.png)

Les explications locales rendent cette dépendance visible : la fraude la mieux notée et la
**fausse alerte** la mieux notée s'expliquent exactement par les deux mêmes facteurs (loisir +
dommage majeur), et la fraude manquée est un sinistre *Minor Damage* avec un loisir ordinaire.
SHAP décrit le modèle, pas le monde : aucune de ces attributions n'est une affirmation causale.

Figures complémentaires : [dépendance SHAP](results/figures/real_shap_dependence.png),
[importance par permutation](results/figures/real_permutation_importance.png),
[distributions EDA](results/figures/eda_numeric_distributions.png),
[matrice de corrélation](results/figures/eda_correlation_matrix.png),
[qualité des données](results/figures/eda_data_quality.png).

## Limites et remarques honnêtes

* **Petit jeu réel.** 800 sinistres d'entraînement et 200 de test (49 fraudes de test). Les
  métriques de test portent une incertitude de ±0,1 sur la PR-AUC ; le classement des modèles
  du groupe de tête n'est pas fiable.
* **Réalisme discutable du benchmark.** La provenance du jeu n'est pas documentée et l'effet
  des loisirs suggère qu'il est en partie simulé. Les résultats décrivent ce benchmark, pas de
  vrais portefeuilles d'assurance.
* **Les données synthétiques n'apportent pas d'information.** Des générateurs appris sur 800
  lignes reproduisent (une partie de) ce que contiennent ces 800 lignes. Les scores synthétiques
  sont rapportés à part et ne mesurent jamais une performance réelle ; les modèles entraînés sur
  données synthétiques ne font pas mieux sur les vrais sinistres que ceux entraînés sur les 800
  sinistres réels, quel que soit le nombre de lignes synthétiques.
* **Les coûts sont des hypothèses.** 1 000 USD par enquête et 10 000 USD par fraude manquée sont
  des valeurs rondes illustratives, pas des données d'assureur ; le tableau de sensibilité
  montre à quel point la conclusion en dépend. Le modèle ignore aussi le montant en jeu propre à
  chaque sinistre.
* **Sélection en CV, évaluation unique sur le test.** Le modèle de référence a été choisi en CV
  avant l'ouverture du test, mais avec 2 à 4 % d'écart entre les meilleurs modèles, une autre
  graine pourrait les classer différemment.
* **Étiquettes de fraude.** `fraud_reported` est le résultat d'un processus d'enquête inconnu ;
  le bruit d'étiquetage et le biais de sélection (quels sinistres ont été enquêtés) sont
  inconnus.
* **Pas de validation en production.** Pas de validation temporelle (les incidents couvrent
  deux mois), pas de suivi de dérive, pas d'audit d'équité (sexe, études et profession sont des
  entrées), pas de test de latence ou de charge de l'API au-delà des tests unitaires.
* **Budget.** 40 essais Optuna par modèle sur un processeur de portable à 6 cœurs. Un budget
  plus grand pourrait déplacer certains modèles de quelques points ; il ne changerait pas la
  conclusion qu'ils sont statistiquement indiscernables sur ces données.

## Reproduire

```bash
git clone https://github.com/fleury-fcn/prediction-sinistres-machine-learning-.git
cd prediction-sinistres-machine-learning-
python3.12 -m venv .venv && source .venv/bin/activate
make install          # dépendances épinglées (Python 3.12)
make all              # eda → train → evaluate → synth, puis exécute les notebooks
make test lint        # tests unitaires (dont fuite et application), ruff + black
```

* `python -m src.main --stage {eda,train,evaluate,synth,all} [--fast]` ; `--fast` exécute tout
  le pipeline avec de tout petits budgets en quelques minutes (test de fumée uniquement, jamais
  pour rapporter des résultats).
* `--cost-fn 20000 --cost-investigation 500` relance l'analyse de coût avec d'autres
  hypothèses ; `--synthetic-rows 1000000` change la taille du jeu synthétique.
* Les graines sont fixées (`seed: 42` dans `configs/default.yaml`) ; l'exécution complète a pris
  environ 1 h 40 sur un Intel Core i7-9750H (6 cœurs). Les runs sont aussi enregistrés dans un
  MLflow local (`mlruns/`, `mlflow ui`), la source de vérité versionnée restant
  `results/metrics/`.
* Note macOS : PyTorch embarque son propre runtime OpenMP, qui se bloque avec celui de
  LightGBM/XGBoost quand les deux tournent en multi-thread dans le même processus. Le code
  limite donc torch à un thread et ne l'importe jamais avant les bibliothèques de boosting.

## Structure du dépôt

```text
├── configs/default.yaml        graines, CV, budget Optuna, coûts, tailles synthétiques (+ mode fast)
├── data/
│   ├── insurance_claims.csv    benchmark réel (1 000 sinistres)
│   └── synthetic/              SYNTHETIC_sample_1000.csv (versionné) ; parquet 500k régénéré
├── src/
│   ├── config.py               chemins, schéma des colonnes (RENAME_MAP), chargement de la config
│   ├── data.py                 chargement, nettoyage déterministe, split
│   ├── features.py             construction de variables (sans état)
│   ├── pipelines.py            prétraitement sans fuite (+ SMOTE)
│   ├── models.py               registre des modèles et espaces de recherche Optuna
│   ├── nn.py                   MLP tabulaire PyTorch (API sklearn)
│   ├── training.py             optimisation Optuna, CV répétée
│   ├── evaluation.py           métriques, bootstrap, coûts, tests statistiques
│   ├── interpret.py            SHAP, importance par permutation
│   ├── synthetic.py            générateurs et contrôles qualité
│   ├── plots.py                style et fonctions de figures
│   ├── stages/                 eda, train, evaluate, synth
│   └── main.py                 CLI
├── notebooks/                  01_eda.ipynb, 02_results.ipynb (exécutés, sorties visibles)
├── results/
│   ├── figures/                toutes les figures présentées ici
│   ├── metrics/                tous les chiffres présentés ici (CSV / JSON / Markdown)
│   └── models/                 pipeline CatBoost exporté + fiche modèle
├── app/                        application Streamlit (FR/EN) + son requirements.txt léger
├── api/app.py                  service FastAPI
├── tests/                      tests pytest
├── Dockerfile                  image de l'API (sans torch)
└── Makefile, pyproject.toml, requirements*.txt, .github/workflows/ci.yml
```

## Servir le modèle

```bash
make api                                   # uvicorn sur :8000
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/claim.json
# {"fraud_probability": …, "flag_for_investigation": …, "decision_threshold": 0.15, "model": "CatBoost"}

make docker-build docker-run               # même API dans une image légère (sans torch)
make app                                   # application Streamlit (FR/EN) : résultats, simulateur, évaluation
```

Le modèle exporté est le meilleur modèle en CV qui n'a pas besoin de PyTorch à l'inférence
(ici CatBoost), sauvegardé avec une [fiche modèle](results/models/model_card.json) contenant
ses paramètres, son seuil, les hypothèses de coût et ses métriques de CV et de test.

## Perspectives

* Valider sur un vrai portefeuille de sinistres plus grand et daté (split temporel, suivi de
  dérive).
* Coûts dépendant de chaque sinistre (montant en jeu) plutôt qu'un coût fixe par fraude
  manquée.
* Audit d'équité et variante sans attributs protégés ou artificiels (sexe, loisirs).
* Prédiction conforme ou calibration de Venn–Abers pour donner aux enquêteurs une incertitude
  par sinistre.
* Tester FT-Transformer / TabNet uniquement une fois davantage de données réelles disponibles.

## Auteur

**Fleury Niyokwizera** · Master 1 MIASHS (Data Modeling, parcours IA), Université de Lille ·
[LinkedIn](https://www.linkedin.com/in/fleury-niyokwizera-2a9436291/)
