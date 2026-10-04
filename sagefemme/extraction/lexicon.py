"""Domain vocabulary used to snap noisy OCR of categorical handwriting to canonical terms.

This is general maternal-health / Moroccan registry vocabulary (not copied from test labels):
abbreviations midwives write (RAS = rien à signaler), exam findings, education levels,
common occupations, the 12 administrative regions, etc. A field-specific list narrows choices.
"""
from __future__ import annotations

from ..schema import _lev, strip_accents

COMMON = ["RAS", "Néant", "Aucun", "Aucune", "Normal", "Normale", "Normales", "Normaux", "Anormal", "Oui", "Non",
          "Non fait", "Neg", "Pos", "Négatif", "Positif"]
FIELD_LEXICON = {
    "education": ["Aucun", "Analphabète", "Primaire", "Collège", "Lycée", "Secondaire", "Supérieur", "Universitaire",
                  "Coranique"],
    "profession": ["Femme au foyer", "Ménagère", "Sans", "Étudiante", "Agricultrice", "Couturière", "Commerçante",
                   "Employée", "Enseignante", "Fonctionnaire", "Ouvrière", "Infirmière", "Coiffeuse", "Artisane"],
    "husband_profession": ["Ouvrier", "Agriculteur", "Fonctionnaire", "Chauffeur", "Maçon", "Mécanicien", "Commerçant",
                           "Employé", "Enseignant", "Militaire", "Sans", "Artisan", "Pêcheur", "Menuisier"],
    "region": ["Tanger-Tétouan-Al Hoceïma", "Oriental", "Fès-Meknès", "Rabat-Salé-Kénitra", "Béni Mellal-Khénifra",
               "Casablanca-Settat", "Marrakech-Safi", "Drâa-Tafilalet", "Souss-Massa", "Guelmim-Oued Noun",
               "Laâyoune-Sakia El Hamra", "Dakhla-Oued Ed-Dahab"],
    "family": ["RAS", "Père", "Mère", "Frère", "Sœur", "Oncle", "Tante", "Grand-père", "Grand-mère", "Cousin", "Cousine"],
    "hx": ["RAS", "Aucun", "Asthme", "Asthme léger", "Anémie", "HTA", "Diabète", "Épilepsie", "Cardiopathie",
           "Appendicectomie", "Césarienne", "Cholécystectomie", "Cycles réguliers", "Cycles irréguliers", "Dysménorrhée"],
    "delivery_mode": ["Voie basse", "Césarienne", "Forceps", "Ventouse", "Instrumental"],
    "cs_indication": ["Souffrance fœtale", "SFA", "Dystocie", "Présentation siège", "Utérus cicatriciel",
                      "Pré-éclampsie sévère", "Placenta prævia", "Disproportion fœto-pelvienne", "Échec de travail"],
    "complication": ["RAS", "Aucune", "Hémorragie", "Infection", "Pré-éclampsie", "Éclampsie", "Déchirure", "Prématurité"],
    "conjunctiva": ["Normales", "Pâles", "Décolorées", "Colorées"],
    "breasts": ["Normaux", "Anormaux"],
    "cervix": ["Fermé", "Ouvert", "Long", "Court", "Mi-long", "Effacé"],
    "presentation": ["Céphalique", "Siège", "Transverse"],
    "exam": ["RAS", "Normal", "Anormal", "Leucorrhées"],
    "place": ["Maternité", "Hôpital", "Domicile", "Centre de santé", "Clinique"],
    "scar": ["Propre", "Propre, sèche", "Infectée", "Suintante"],
    "decision": ["Poursuivre l'allaitement exclusif", "Référer", "Contrôle", "RAS"],
    "nb_anomaly": ["Aucune", "RAS", "Pied bot", "Fente labiale"],
    "sex": ["F", "M"],
    "screening": ["Non fait", "Normal", "Anormal", "Fait"],
    "staff": ["Sage-femme", "Inf.", "Dr"],
}


def lexicon_for(fid: str) -> list[str]:
    if fid in ("education", "profession", "husband_profession", "region"):
        return FIELD_LEXICON[fid]
    if fid.startswith("fh_"):
        return FIELD_LEXICON["family"]
    if fid.startswith("hx_"):
        return FIELD_LEXICON["hx"] + COMMON
    if fid.endswith("_mode"):
        return FIELD_LEXICON["delivery_mode"]
    if fid.endswith("cs_indication"):
        return FIELD_LEXICON["cs_indication"]
    if "complication" in fid:
        return FIELD_LEXICON["complication"] + COMMON
    if fid.startswith("g_conjunctiva"):
        return FIELD_LEXICON["conjunctiva"]
    if fid.startswith("g_breasts"):
        return FIELD_LEXICON["breasts"]
    if fid.startswith("g_cervix"):
        return FIELD_LEXICON["cervix"]
    if fid.startswith("g_presentation"):
        return FIELD_LEXICON["presentation"]
    if fid.startswith(("g_skeleton", "g_speculum", "g_pelvis")):
        return FIELD_LEXICON["exam"]
    if fid.startswith("ob_") and fid.endswith("_place"):
        return FIELD_LEXICON["place"]
    if fid.endswith("_scar"):
        return FIELD_LEXICON["scar"]
    if fid.endswith("decision"):
        return FIELD_LEXICON["decision"]
    if fid == "nb_anomaly":
        return FIELD_LEXICON["nb_anomaly"]
    if fid == "nb_sex":
        return FIELD_LEXICON["sex"]
    if fid == "cervical_screening":
        return FIELD_LEXICON["screening"]
    return COMMON


def _k(s: str) -> str:
    return strip_accents(s).lower().replace(" ", "").replace("'", "").replace("-", "").replace(".", "").replace(",", "")


def snap(fid: str, text: str) -> tuple[str, float]:
    """Return (canonical_text, similarity 0..1). Similarity 1.0 = exact (accent-insensitive)."""
    if not text:
        return text, 0.0
    k = _k(text)
    if not k:
        return text, 0.0
    best, bd = None, 1e9
    second = 1e9
    for cand in lexicon_for(fid):
        d = _lev(k, _k(cand))
        if d < bd:
            best, second, bd = cand, bd, d
        elif d < second:
            second = d
    L = max(len(k), len(_k(best)))
    sim = 1 - bd / L
    tol = 0 if L <= 2 else (1 if L <= 5 else max(2, int(L * 0.25)))
    if bd <= tol and (second > bd or bd == 0):
        return best, sim
    return text, 0.0
