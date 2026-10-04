"""Field schema derived from the real registry pages (8 page types, see specimen PDF).

Every field is defined by *where it lives on the printed form* (an anchor label printed in
Helvetica + a geometric rule), its value type, and its importance. The same definition is used
  - to extract ground truth from the specimen PDF text layer (tools/build_dataset.py),
  - to build page templates (regions in PDF points on a reference page),
  - by the local OCR backend (regions projected onto the photo through a homography),
  - to generate the per-page JSON schema given to the vision-LLM backend.

Kinds
  inline  : handwritten value to the right of a printed label on the same line
  cell    : table cell at (row label, column header)
  below   : free-text box under a header
  check   : a checkbox next to a printed label (value = ticked or not)
  checkrow: the i-th checkbox to the right of a label (e.g. "VAT : [1] [2] [3]...")
  pii     : a direct identifier (name, CIN, phone, address...). Located ONLY to be redacted
            from the image before any processing. Never extracted, never stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

PAGE_TYPES = {
    # id: (title keywords for classification, FR label, EN label)
    "cover": (("fiche", "surveillance"), "Fiche de surveillance", "Cover sheet"),
    "identification": (("identification", "antecedents"), "Identification et antécédents", "Identification & history"),
    "pregnancy": (("grossesse", "actuelle"), "Grossesse actuelle", "Current pregnancy"),
    "delivery": (("deroulement", "accouchement"), "Déroulement de l'accouchement", "Delivery"),
    "pp_mother_early": (("precoce", "mere"), "Post-partum précoce — mère", "Early postpartum — mother"),
    "pp_newborn_early": (("precoce", "nouveau"), "Post-partum précoce — nouveau-né", "Early postpartum — newborn"),
    "pp_mother_late": (("tardif", "mere"), "Post-partum tardif — mère", "Late postpartum — mother"),
    "pp_newborn_late": (("tardif", "nouveau"), "Post-partum tardif — nouveau-né", "Late postpartum — newborn"),
}
PAGE_ORDER = list(PAGE_TYPES)


@dataclass
class Spec:
    id: str
    page: str
    kind: str
    fr: str
    type: str = "text"                  # text|int|float|date|bool|bp|ga|lab|code
    anchor: str = ""
    nth: int = 0
    w: float = 120                      # inline / below width (pt)
    h: float = 0                        # below height / cell half-height (pt)
    dx0: float = 1.0                    # inline: gap after label
    row: str = ""                       # cell
    row_nth: int = 0
    col: str = ""
    col_nth: int = 0
    col_hw: float = 22                  # cell half-width (pt)
    side: str = "left"                  # check: where the box is relative to its label
    idx: int = 0                        # checkrow index
    exact: bool = False                 # anchor must equal the whole printed run
    importance: int = 1                 # 3 = critical (linking / key clinical), 2 = important
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    group: str = ""                     # display group inside a page
    en: str = ""


S: list[Spec] = []


def add(*a, **k):
    S.append(Spec(*a, **k))


def inline(fid, page, fr, anchor, w=120, t="text", **k):
    add(fid, page, "inline", fr, t, anchor=anchor, w=w, **k)


def check(fid, page, fr, anchor=None, side="left", **k):
    add(fid, page, "check", fr, "bool", anchor=anchor or fr, side=side, exact=True, **k)


def pii(fid, page, anchor, w=200, **k):
    add(fid, page, "pii", "IDENTIFIANT DIRECT (masqué)", "text", anchor=anchor, w=w, **k)


# ------------------------------------------------------------------ cover sheet
P = "cover"
inline("fiche_number", P, "N° de la fiche", "N° de la fiche :", 170, "code", importance=3, group="Fiche",
       en="Registry number (linking code)")
inline("region", P, "Région", "Région :", 190, group="Fiche", en="Region")
inline("province", P, "Province", "Province :", 160, group="Fiche", en="Province")
inline("facility", P, "Établissement sanitaire", "Nom de l'établissement sanitaire :", 260, group="Fiche",
       en="Health facility")
for a in ("DR", "CSC", "CSU", "CSCA", "CSUA"):
    check(f"fac_{a.lower()}", P, f"Type établissement : {a}", a, side="right", group="Fiche", en=f"Facility type {a}")
check("fac_fixed", P, "Fixe", "Fixe", group="Fiche", en="Fixed facility")
check("fac_mobile", P, "Mobile", "Mobile", group="Fiche", en="Mobile unit")
pii("pii_name", P, "Nom/Prénom de la parturiente :", 260)
check("risk", P, "Grossesse classée à risque", "Grossesse classée à risque :", group="Risque", en="Pregnancy classified at risk")
for fid, a in (("risk_anemia", "Anémie"), ("risk_hta", "H.T.A"), ("risk_diabetes", "Diabète"),
               ("risk_cardio", "Cardiopathie"), ("risk_metro", "Métrorragie"), ("risk_infection", "Infection"),
               ("risk_preeclampsia", "Pré-éclampsie"), ("risk_eclampsia", "Eclampsie")):
    check(fid, P, f"Risque : {a}", a, group="Risque", en=f"Risk: {a}")
inline("risk_other", P, "Risque : autres", "Autres à préciser :", 150, group="Risque", en="Risk: other")

# ------------------------------------------------------------------ identification & history
P = "identification"
inline("age", P, "Âge", "Age :", 110, "int", vmin=12, vmax=55, importance=3, group="Profil", en="Age")
inline("education", P, "Niveau d'instruction", "Niveau d'instruction :", 110, group="Profil", en="Education")
inline("profession", P, "Profession", "Profession :", 140, nth=0, group="Profil", en="Occupation")
inline("husband_profession", P, "Profession du mari", "Profession :", 140, nth=1, group="Profil",
       en="Husband's occupation")
pii("pii_cin", P, "CIN :", 150)
pii("pii_address", P, "Adresse :", 160)
pii("pii_phone", P, "Téléphone :", 150)
pii("pii_husband", P, "Nom du Mari :", 150)
check("consanguinity", P, "Consanguinité", group="Profil", importance=2, en="Consanguinity")
check("desired_pregnancy", P, "Grossesse désirée", group="Profil", en="Desired pregnancy")
for rid, rl in (("hta", "HTA"), ("diabetes", "Diabète"), ("hereditary", "Maladies héréditaires"),
                ("malformations", "Malformations"), ("allergies", "Allergie(s)")):
    for cid, cl in (("woman", "Famille de la femme"), ("husband", "Mari/famille")):
        add(f"fh_{rid}_{cid}", P, "cell", f"{rl} — {cl}", "text", row=rl, col=cl, col_hw=40, h=9,
            exact=True, group="Antécédents familiaux", en=f"Family history {rl} ({cid})")
for fid, a, fr in (("hx_medical", "Médicaux", "Antécédents médicaux"), ("hx_surgical", "Chirurgicaux", "Antécédents chirurgicaux"),
                   ("hx_gyneco", "Gynécologiques", "Antécédents gynécologiques")):
    add(fid, P, "below", fr, "text", anchor=a, w=72 if a != "Gynécologiques" else 105, h=40, exact=True,
        importance=2, group="Antécédents de la femme", en=fr)
for rid, rl in (("abortion", "Avortement"), ("premature", "Accouchement prématuré"),
                ("iufd", "Mort fœtale in utéro"), ("other", "Autres à préciser :")):
    for cid, cl, t, hw in (("n", "Nombre", "int", 30), ("date", "Date", "text", 50), ("place", "Lieu", "text", 40),
                           ("ga", "Age gestationnel (SA)", "ga", 45)):
        add(f"ob_{rid}_{cid}", P, "cell", f"{rl} — {cl}", t, row=rl, col=cl, col_hw=hw, h=10,
            importance=2 if cid == "n" else 1, group="Antécédents obstétricaux", en=f"Obstetric history {rid} {cid}")
for k in range(1, 6):
    for rid, rl, rn, t in (("date", "Date", 1, "date"), ("mode", "Modalité d'extraction", 0, "text"),
                           ("cs_indication", "Si césarienne : indication", 0, "text"),
                           ("complication", "Complication (type)", 0, "text"),
                           ("weight", "Poids nouveau-né(s)", 0, "int"),
                           ("nb_complication", "Compl. nouveau-né (type)", 0, "text")):
        add(f"pd{k}_{rid}", P, "cell", f"Accouch. {k} — {rl}", t, row=rl, row_nth=rn, col=f"Accouch. {k}",
            col_hw=38, h=13, vmin=400 if t == "int" else None, vmax=6000 if t == "int" else None,
            group="Accouchements antérieurs", en=f"Previous delivery {k} {rid}")
inline("gravidity", P, "Gestation", "Gestation :", 45, "int", vmin=1, vmax=20, importance=2, group="Obstétrique",
       en="Gravidity")
inline("parity", P, "Parité", "Parité :", 45, "int", vmin=0, vmax=20, importance=2, group="Obstétrique", en="Parity")
inline("living_children", P, "Nombre d'enfants vivants", "Nombre d'enfants vivants :", 45, "int", vmin=0, vmax=20,
       group="Obstétrique", en="Living children")
for i in range(1, 6):
    add(f"vat_{i}", P, "checkrow", f"VAT dose {i}", "bool", anchor="VAT :", idx=i - 1, group="Vaccination",
        en=f"Tetanus toxoid dose {i}")
check("rubella_vacc", P, "Vaccinée contre la rubéole", group="Vaccination", en="Rubella vaccinated")
inline("rubella_vacc_date", P, "Rubéole — date", "Le", 90, "date", nth=0, exact=True, group="Vaccination",
       en="Rubella vaccination date")
check("hepb_vacc", P, "Vaccinée contre l'hépatite B", group="Vaccination", en="Hepatitis B vaccinated")
inline("hepb_vacc_date", P, "Hépatite B — date", "Le", 90, "date", nth=1, exact=True, group="Vaccination",
       en="Hepatitis B vaccination date")
inline("cervical_screening", P, "Frottis cervical / IVA", "Frottis cervical / IVA (moins de 3 ans) :", 110,
       group="Vaccination", en="Cervical screening")

# ------------------------------------------------------------------ current pregnancy
P = "pregnancy"
inline("lmp", P, "DDR", "DDR :", 75, "date", importance=2, group="Datation", en="Last menstrual period")
inline("height_cm", P, "Taille (cm)", "Taille :", 60, "int", vmin=120, vmax=200, group="Datation", en="Height (cm)")
inline("edd", P, "Date prévue d'accouchement", "DATE PRÉVUE D'ACCOUCHEMENT :", 80, "date", importance=3,
       group="Datation", en="Expected delivery date")
inline("term_date", P, "Date de dépassement de terme", "DATE DE DÉPASSEMENT DE TERME :", 80, "date",
       group="Datation", en="Post-term date")
for i, (fid, fr) in enumerate((("bg_a", "Groupe A"), ("bg_b", "Groupe B"), ("bg_o", "Groupe O"), ("bg_ab", "Groupe AB"),
                               ("rh_neg", "Rh-"))):
    add(fid, P, "checkrow", fr, "bool", anchor="Groupage :", idx=i, group="Groupage", en=fr)
check("rh_pos", P, "Rh+", group="Groupage", en="Rh+")

VISIT_COLS = [("t1v1", "Visite 1", 0, "T1 V1"), ("t1v2", "Visite 2", 0, "T1 V2"), ("t1v3", "Visite 3", 0, "T1 V3"),
              ("t2v1", "Visite 1", 1, "T2 V1"), ("t2v2", "Visite 2", 1, "T2 V2"), ("t2v3", "Visite 3", 1, "T2 V3"),
              ("m7", "7ème mois", 0, "7e mois"), ("m8", "8ème mois", 0, "8e mois"), ("m9", "9ème mois", 0, "9e mois")]
GRID_ROWS = [
    ("rdv", "Rendez-vous", "date", 0, "Appointment"), ("visit_date", "Venue le", "date", 2, "Visit date"),
    ("reminder", "Visites de relance", "bool", 0, "Reminder visit"), ("ga", "Age probable", "ga", 2, "Gestational age"),
    ("weight", "Poids (kg)", "float", 2, "Weight (kg)"), ("bp", "TA", "bp", 3, "Blood pressure"),
    ("skeleton", "Anomalies squelette", "text", 0, "Skeletal anomalies"),
    ("conjunctiva", "État des conjonctives", "text", 1, "Conjunctivae"),
    ("breasts", "Examen des seins", "text", 0, "Breast exam"), ("edema", "Œdèmes", "bool", 2, "Oedema"),
    ("fetal_mvt", "Mouvements actifs", "bool", 1, "Fetal movements"), ("fundal_height", "HU (cm)", "int", 1, "Fundal height"),
    ("fhr", "BCF", "int", 2, "Fetal heart rate"), ("speculum", "Examen au spéculum", "text", 0, "Speculum exam"),
    ("cervix", "TV : état du col", "text", 0, "Cervix"), ("presentation", "TV : présentation", "text", 1, "Presentation"),
    ("pelvis", "TV : bassin", "text", 0, "Pelvis"), ("glycosuria", "Glucosurie", "lab", 1, "Glycosuria"),
    ("albuminuria", "Albuminurie", "lab", 2, "Albuminuria"), ("rubella", "Rubéole", "lab", 1, "Rubella serology"),
    ("toxo", "Toxoplasmose", "lab", 1, "Toxoplasmosis"), ("syphilis", "Syphilis (TPHA/VDRL)", "lab", 3, "Syphilis"),
    ("hbsag", "Ag HBs", "lab", 2, "HBs antigen"), ("hiv", "Sérologie VIH", "lab", 3, "HIV"),
    ("hb", "Hémoglobine", "float", 2, "Haemoglobin (g/dL)"), ("platelets", "Plaquettes", "text", 1, "Platelets"),
    ("glycemia", "Bilan glycémique", "float", 2, "Glycaemia (g/L)"), ("rai", "RAI (si Rh négatif)", "lab", 1, "RAI"),
    ("iron", "Fer", "bool", 1, "Iron"), ("examiner", "Examen fait par", "text", 0, "Examined by"),
]
RANGES = {"weight": (30, 150), "fundal_height": (5, 45), "fhr": (80, 200), "hb": (4, 20), "glycemia": (0.3, 4)}
for rid, rl, t, imp, en in GRID_ROWS:
    for cid, cl, cn, cen in VISIT_COLS:
        lo, hi = RANGES.get(rid, (None, None))
        add(f"g_{rid}_{cid}", P, "cell", f"{rl} — {cen}", t, row=rl, row_nth=1 if rid == "examiner" else 0,
            col=cl, col_nth=cn, col_hw=22.5, h=8.5, exact=True, importance=imp, vmin=lo, vmax=hi,
            group=rl, en=f"{en} — {cen}")

# ------------------------------------------------------------------ delivery
P = "delivery"
pii("pii_patient", P, "Patiente :", 240)
check("del_supervised", P, "En milieu surveillé", group="Lieu", en="Supervised setting")
check("del_birth_home", P, "Maison d'accouchement", group="Lieu", en="Birthing home")
check("del_maternity", P, "Maternité", group="Lieu", en="Maternity")
check("del_clinic", P, "Clinique privée", group="Lieu", en="Private clinic")
inline("del_place_other", P, "Lieu : autres", "Autres :", 200, nth=0, group="Lieu", en="Place: other")
check("del_home", P, "A domicile", group="Lieu", en="At home")
check("del_assisted", P, "Assisté par un personnel qualifié", group="Lieu", en="Skilled attendant")
inline("del_home_other", P, "Domicile : autres", "Autres :", 200, nth=1, group="Lieu", en="Home: other")
inline("del_date", P, "Date de l'accouchement", "Date de l'accouchement :", 100, "date", importance=3, group="Accouchement",
       en="Delivery date")
for fid, a, en in (("mode_vaginal", "Voie basse non instrumentale", "Spontaneous vaginal"),
                   ("mode_instrumental", "Voie basse instrumentale", "Instrumental vaginal"),
                   ("mode_forceps", "Forceps", "Forceps"), ("mode_vacuum", "Ventouse", "Vacuum"),
                   ("mode_episiotomy", "Avec épisiotomie", "Episiotomy"),
                   ("mode_cs_planned", "Césarienne : Programmée", "Planned caesarean"),
                   ("mode_cs_emergency", "Urgence", "Emergency caesarean")):
    check(fid, P, a, group="Mode", importance=2, en=en)
inline("cs_indication", P, "Indication de la césarienne", "Préciser l'indication :", 200, group="Mode",
       en="Caesarean indication")
check("del_complications", P, "Présence de complications", side="right", importance=2, group="Complications",
      en="Complications present")
for fid, a, en in (("dc_at_delivery", "Au moment de l'accouchement", "At delivery"), ("dc_postpartum", "Suites de couches", "Postpartum"),
                   ("dc_preeclampsia", "Pré-éclampsie", "Pre-eclampsia"), ("dc_eclampsia", "Eclampsie", "Eclampsia"),
                   ("dc_haemorrhage", "Hémorragie", "Haemorrhage"), ("dc_infection", "Infection", "Infection")):
    check(fid, P, a, group="Complications", en=en)
check("dc_other", P, "Autres", "Autres", nth=0, group="Complications", en="Other")
inline("dc_other_text", P, "Complications : préciser", "Si autres à préciser :", 120, group="Complications",
       en="Other complication")
check("nb_alive", P, "Vivant", importance=3, group="Nouveau-né", en="Born alive")
check("nb_stillborn", P, "Mort-né", importance=3, group="Nouveau-né", en="Stillborn")
check("nb_death24", P, "Décès < 24 heures", importance=3, group="Nouveau-né", en="Death < 24h")
inline("nb_sex", P, "Sexe", "Sexe :", 60, group="Nouveau-né", importance=2, en="Sex")
inline("nb_weight", P, "Poids à la naissance (g)", "Poids à la naissance :", 90, "int", vmin=400, vmax=6500, importance=3,
       group="Nouveau-né", en="Birth weight (g)")
inline("nb_hc", P, "Périmètre crânien (cm)", "Périmètre crânien à la naissance :", 70, "float", vmin=20, vmax=45,
       group="Nouveau-né", en="Head circumference (cm)")
inline("nb_anomaly", P, "Anomalie", "Anomalie à préciser :", 200, group="Nouveau-né", en="Anomaly")
inline("nb_ga", P, "Âge gestationnel", "Âge gestationnel :", 60, "ga", importance=2, group="Nouveau-né",
       en="Gestational age at birth")


# ------------------------------------------------------------------ postpartum mother (early / late)
def pp_mother(P, pre, early):
    a1, a2 = (("Entre le 7ème et 8ème jour après l'accouchement", "Après le 8ème jour de l'accouchement") if early else
              ("Entre le 40ème et 50ème jour après l'accouchement", "Après le 50ème jour de l'accouchement"))
    pii(f"{pre}pii_header", P, "MÈRE —", 0)
    check(f"{pre}timing_window", P, "Dans la fenêtre recommandée", a1, side="right", group="Consultation",
          en="Within recommended window")
    check(f"{pre}timing_after", P, "Après la fenêtre", a2, side="right", group="Consultation", en="After window")
    inline(f"{pre}date", P, "Date de la consultation", "Date de la consultation :", 90, "date", importance=2,
           group="Consultation", en="Consultation date")
    inline(f"{pre}temp", P, "T° (°C)", "T°", 40, "float", vmin=34, vmax=42, importance=2, group="Signes vitaux",
           en="Temperature")
    inline(f"{pre}bp", P, "TA", "TA", 70, "bp", exact=True, importance=3, group="Signes vitaux", en="Blood pressure")
    inline(f"{pre}pulse", P, "Pouls", "Pouls", 40, "int", vmin=30, vmax=200, group="Signes vitaux", en="Pulse")
    inline(f"{pre}weight", P, "Poids (kg)", "Poids", 50, "float", vmin=30, vmax=150, group="Signes vitaux", en="Weight")
    for fid, a, n, g in (("conj_normal", "Normales", 0, "Conjonctives"), ("conj_pale", "Décolorées", 0, "Conjonctives"),
                         ("uterine_globe", "Présence du globe utérin", 0, "Utérus"),
                         ("lochia_bland", "Fade", 0, "Lochies"), ("lochia_fetid", "fétide", 0, "Lochies"),
                         ("lochia_clear", "claires", 0, "Lochies"), ("lochia_bloody", "sanglantes", 0, "Lochies"),
                         ("lochia_yellow", "Jaunâtres", 0, "Lochies"),
                         ("perineum_normal", "Normal", 0, "Périnée"), ("perineum_episiotomy", "Épisiotomie", 0, "Périnée"),
                         ("perineum_repaired", "Réparée", 0, "Périnée"), ("perineum_tear", "Déchirure", 0, "Périnée"),
                         ("sphincter_normal", "Normal", 1, "Sphincters"), ("sphincter_abnormal", "Anormal", 0, "Sphincters"),
                         ("cesarean", "Césarienne :", 0, "Césarienne"),
                         ("breast_normal", "Normal", 2, "Seins"), ("breast_lymphangitis", "lymphangite", 0, "Seins"),
                         ("breast_mastitis", "mastite et abcès", 0, "Seins"),
                         ("calf_normal", "Normal", 3, "Mollets"), ("calf_red", "Rouges", 0, "Mollets"),
                         ("calf_warm", "Chauds", 0, "Mollets"), ("calf_pain", "Douloureux à la dorsiflexion", 0, "Mollets")):
        check(f"{pre}{fid}", P, f"{g} : {a}", a, nth=n, group=g, en=f"{g}: {a}")
    inline(f"{pre}scar", P, "État de la cicatrice", "Etat de la cicatrice :", 150, group="Césarienne", en="Scar")
    check(f"{pre}complication", P, "Présence de complication", "Présence de complication :", side="right",
          importance=2, group="Complications", en="Complication present")
    for fid, a in (("c_haemorrhage", "Hémorragie"), ("c_infection", "Infection"), ("c_eclampsia", "Eclampsie"),
                   ("c_phlebitis", "Phlébite"), ("c_breast", "Complications mammaires"), ("c_anemia", "Anémie"),
                   ("c_other", "Autres")):
        check(f"{pre}{fid}", P, f"Complication : {a}", a, group="Complications", en=f"Complication: {a}")
    check(f"{pre}medication", P, "Prise de médicaments", "Notion de prise de médicaments :", side="right",
          group="Traitement", en="Taking medication")
    inline(f"{pre}medication_text", P, "Médicaments (précision)", "Notion de prise de médicaments :", 300, dx0=100,
           group="Traitement", en="Medication details")
    check(f"{pre}tx_iron", P, "Traitement : Fer", "Fer", group="Traitement", en="Iron prescribed")
    check(f"{pre}tx_vita", P, "Traitement : Vitamine A", "Vitamine A", group="Traitement", en="Vitamin A prescribed")
    inline(f"{pre}tx_other", P, "Traitement : autres", "Autres à préciser :", 180, group="Traitement", en="Other treatment")
    inline(f"{pre}next_visit", P, "Prochain rendez-vous", "Prochain rendez-vous le", 100, "date", group="Traitement",
           en="Next appointment")
    check(f"{pre}fp_wants", P, "PF : désire une méthode", "Désire utiliser une méthode", group="Planification familiale",
          en="FP: wants a method")
    check(f"{pre}fp_pill", P, "PF : pilule", "pilule", group="Planification familiale", en="FP: pill")
    check(f"{pre}fp_iud", P, "PF : DIU", "DIU", group="Planification familiale", en="FP: IUD")
    inline(f"{pre}fp_other", P, "PF : autre méthode", "Autre à préciser :", 130, group="Planification familiale",
           en="FP: other method")
    check(f"{pre}fp_prescribed", P, "PF : prescription faite", "Prescription faite", group="Planification familiale",
          en="FP: prescribed")
    check(f"{pre}fp_referred", P, "PF : référée", "Référée :", group="Planification familiale", en="FP: referred")
    add(f"{pre}fp_why_not", P, "below", "PF : pourquoi pas de méthode", "text",
        anchor="Si la mère ne désire pas une méthode contraceptive : Pourquoi ?", w=480, h=26,
        group="Planification familiale", en="FP: reason for no method")


def pp_newborn(P, pre):
    inline(f"{pre}date", P, "Date de la consultation", "Date de la consultation :", 90, "date", importance=2,
           group="Consultation", en="Consultation date")
    inline(f"{pre}age_days", P, "Âge (jours)", "Age", 55, "int", exact=True, vmin=0, vmax=90, group="Mesures",
           en="Age (days)")
    inline(f"{pre}temp", P, "Température (°C)", "Température", 55, "float", vmin=33, vmax=42, importance=2,
           group="Mesures", en="Temperature")
    inline(f"{pre}weight", P, "Poids (g)", "Poids", 60, "int", vmin=400, vmax=8000, importance=2, group="Mesures",
           en="Weight (g)")
    inline(f"{pre}length", P, "Taille (cm)", "Taille", 50, "int", vmin=30, vmax=70, group="Mesures", en="Length (cm)")
    inline(f"{pre}hc", P, "Périmètre crânien (cm)", "Périmètre crânien", 50, "float", vmin=25, vmax=50,
           group="Mesures", en="Head circumference")
    for fid, a, n, g in (("premature", "Nouveau-né prématuré", 0, "Mesures"), ("hypotrophic", "Nouveau-né hypotrophe", 0, "Mesures"),
                         ("bf_exclusive", "exclusivement au sein", 0, "Allaitement"),
                         ("bf_artificial", "Artificiel", 0, "Allaitement"), ("bf_mixed", "mixte", 0, "Allaitement"),
                         ("ds_convulsions", "Convulsions", 0, "Signes de danger"), ("ds_feeding", "Refus de téter", 0, "Signes de danger"),
                         ("ds_hematemesis", "Hématémèses", 0, "Signes de danger"), ("ds_melena", "Mélaenas", 0, "Signes de danger"),
                         ("ds_diarrhoea", "Diarrhée", 0, "Signes de danger"), ("ds_jaundice", "Ictère", 0, "Signes de danger"),
                         ("ds_chest_indrawing", "Tirage sous costal", 0, "Signes de danger"), ("ds_cough", "Toux", 0, "Signes de danger"),
                         ("ds_resp_rate", "Rythme respiratoire anormal", 0, "Signes de danger"),
                         ("ds_fever", "Fièvre", 0, "Signes de danger"), ("ds_hypothermia", "Hypothermie", 0, "Signes de danger"),
                         ("tr_cephalhematoma", "Bosse sérosanguine ou céphalohématome", 0, "Traumatismes"),
                         ("tr_hip", "Luxation congénitale de la hanche", 0, "Traumatismes"),
                         ("tr_limb", "Diminution ou absence de la mobilité d'un membre", 0, "Traumatismes"),
                         ("bfe_normal", "Normal", 0, "Évaluation allaitement"), ("bfe_problems", "A problèmes", 0, "Évaluation allaitement"),
                         ("vacc_bcg", "BCG", 0, "Vaccination"), ("vacc_hb", "HB", 0, "Vaccination"),
                         ("vit_d", "Supplémentation en vitamine D", 0, "Vaccination"),
                         ("c_jaundice", "Ictère", 1, "Complications"), ("c_infection", "Infection", 0, "Complications"),
                         ("c_conjunctivitis", "Conjonctivite", 0, "Complications"), ("c_trauma", "Traumatisme", 0, "Complications"),
                         ("c_malformation", "Malformation", 0, "Complications"), ("c_other", "Autres", 0, "Complications")):
        imp = 2 if g == "Signes de danger" else 1
        check(f"{pre}{fid}", P, f"{g} : {a}", a, nth=n, group=g, importance=imp, en=f"{g}: {a}")
    inline(f"{pre}ds_other", P, "Signes de danger : autres", "Autres à préciser :", 200, nth=0, group="Signes de danger",
           en="Other danger signs")
    inline(f"{pre}tr_other", P, "Traumatismes : autres", "Autres à préciser :", 200, nth=1, group="Traumatismes",
           en="Other trauma")
    inline(f"{pre}seen_by", P, "Vu par", "Vu par :", 200, group="Conduite", en="Seen by")
    inline(f"{pre}decision", P, "Décision prise", "Décision prise :", 330, group="Conduite", en="Decision")
    inline(f"{pre}treatment", P, "Traitement prescrit", "Traitement prescrit :", 320, group="Conduite", en="Treatment")
    check(f"{pre}transfer", P, "Transfert", group="Conduite", importance=2, en="Transfer / referral")
    inline(f"{pre}referral_facility", P, "Établissement de référence", "Préciser l'établissement de référence :", 280,
           group="Conduite", en="Referral facility")
    inline(f"{pre}next_visit", P, "Prochaine visite", "Revenir pour une visite de suivi nécessaire le", 100, "date",
           group="Conduite", en="Next visit")


pp_mother("pp_mother_early", "pme_", True)
pp_newborn("pp_newborn_early", "nbe_")
pp_mother("pp_mother_late", "pml_", False)
pp_newborn("pp_newborn_late", "nbl_")

SPECS: list[Spec] = S
SPEC_BY_ID = {s.id: s for s in S}
DATA_SPECS = [s for s in S if s.kind != "pii"]
PII_SPECS = [s for s in S if s.kind == "pii"]
assert len(SPEC_BY_ID) == len(S), "duplicate field id"
