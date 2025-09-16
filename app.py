import os, re, io, json, hashlib
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Initialisation

load_dotenv()
st.set_page_config(page_title="CV Chatbot · Jérôme TAM", page_icon="💼", layout="centered")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Clé API manquante. Ajoute GROQ_API_KEY dans les secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
DEFAULT_MODEL = "llama-3.3-70b-versatile"
CV_TEXT = st.secrets.get("CV_TEXT", "").strip()

MAX_QUESTIONS = 10  



# Thèmes généraux d'entretien (FR/EN)
TECH_WHITELIST = {
    
    "cv","profil","expérience","experiences","projet","projets","compétence","competences","skills",
    "salaire","prétentions","pretentions","forces","faiblesses","disponibilité","mobilité","education",
    "formation","langues","github","linkedin","portfolio","dispo","contrat","stage","alternance",
    "freelance","cdi","cdd","mission",
   
    "groq","streamlit","fastapi","aws","ec2","s3","rds","docker","pytorch","xgboost","lightgbm","prophet",
    "arima","sql","postgresql","llm","rag","lora","quantization","vision","vit","conformer","wav2vec",
    "vq","gumbel","diffusion","unet","kalalodata","activus","rsna","recipeprep","app","ios","kaggle",
    "weaviate","langchain","langgraph","mops","mlops","we","w&b","weights","biases","power bi","tableau"
}

GENERAL_INTERVIEW_TERMS = {
    "soft skills","culture fit","work style","under pressure","teamwork","communication","leadership","ownership",
    "problem solving","problem-solving","conflict","deadline","stress","adaptability","autonomy","collaboration",
    "motivation","learning","feedback","prioritization","time management","stakeholder","mentoring","pair programming","team","work",
    
    "sous pression","travail en équipe","communication","leadership","problème","résolution de problème","conflit",
    "échéance","stress","adaptabilité","autonomie","collaboration","motivation","apprentissage","priorisation",
    "gestion du temps","parties prenantes","encadrement","mentorat","culture","valeurs","éthique","équipe","equipe","travail",
    "temps","gestion","proposition","esprit","integration","Pitch","pitch","candidat","Candidat","Jerome","Jérôme","Peut-il","peut-il",
    "peut il","peut","est ce que","sait il","sait",'problem'
}


TECH_CAPABILITY_TERMS = {
    "mlops","pipeline","ml pipeline","workflow","cicd","monitoring","observability","testing","unit tests",
    "design","architecture","scalability","security","privacy","data quality","feature store",
    "serving","deployment","inference","latency","cost optimization",
  
    "chaîne ml","chaîne de ml","déploiement","surveillance","observabilité","tests","architecture","scalabilité",
    "sécurité","confidentialité","qualité des données","magasin de features","serving","inférence","latence","coût"
}


SOFT_SKILL_PATTERNS = [
    r"bon\s+.*sous\s+pression", r"work(ing)?\s+under\s+pressure",
    r"(soft\s*skills?|compétences\s+comportementales?)",
    r"communication|communicant|communication skills",
    r"(team|équipe).*(player|travail)", r"leadership",
    r"(gestion|management)\s+du\s+temps|time\s+management",
    r"résolution\s+de\s+probl(è|e)me|problem[-\s]*solving",
    r"adaptabilit(é|e)|adaptability", r"motivation|motivated",
]


def extract_keywords(text: str) -> set:
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-\+]{3,}", text.lower())
    freq = {}
    for w in words:
        if w.isdigit():
            continue
        freq[w] = freq.get(w, 0) + 1
    top = {w for (w, c) in freq.items() if c >= 2}
    return top

CV_TERMS = extract_keywords(CV_TEXT)

ALLOWED_TERMS = (
    set(w.lower() for w in TECH_WHITELIST)
    | set(w.lower() for w in GENERAL_INTERVIEW_TERMS)
    | set(w.lower() for w in TECH_CAPABILITY_TERMS)
    | CV_TERMS
)

def always_allow(query: str) -> bool:
    q = query.lower()
   
    for pat in SOFT_SKILL_PATTERNS:
        if re.search(pat, q):
            return True
 
    if ("candidat" in q or "candidate" in q) and any(k in q for k in [
        "bon","good","peut","can","sait","know","capable","able","compétent","competent"
    ]):
        return True

    toks = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-\+]{3,}", q))
    if toks & (GENERAL_INTERVIEW_TERMS | TECH_CAPABILITY_TERMS):
        return True
    return False

GATE_THRESHOLD = 0.05

def is_on_topic(query: str, threshold: float = GATE_THRESHOLD) -> bool:
    if always_allow(query):
        return True
    q_tokens = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-\+]{3,}", query.lower()))
    if not q_tokens:
        return False
    overlap = len(q_tokens & ALLOWED_TERMS) / max(1, len(q_tokens))
    return overlap >= threshold



# System prompt 
def build_system_prompt(cv_text: str, style: str) -> str:
    cv_snippet = cv_text[:6000]  # limiter les tokens
    base = f"""
Tu es un assistant d'entretien qui répond STRICTEMENT à partir du CV ci-dessous (expériences, projets, **formation**).
Langue: FR si l'utilisateur parle FR; sinon EN. Style concis, concret, structuré (bullets OK).

CV du candidat (verbatim) :
---
{cv_snippet}
---

# POLITIQUE DE RÉPONSE
1) On-topic uniquement : profil, compétences, expériences, projets, forces/faiblesses, salaire, dispo, soft skills, MLOps.
2) Zéro hallucination : ne pas inventer d'outils/expériences absents du CV.
3) Hiérarchie des preuves (dans cet ordre) :
   A. Expériences professionnelles / projets livrés
   B. Projets personnels / compétitions (Kaggle, GitHub)
   C. **Formation (cours/TP/outils)** → autorisée pour **inférer** une capacité si pas explicitement mentionnée ailleurs.
4) Questions de type "Le candidat sait-il faire X ?" :
   a) Si preuve explicite (A/B) → **Verdict : Oui/Non** + 2–3 puces d'évidence (citations du CV).
   b) Si pas explicite mais **formation pertinente (C)** :
      • Dis-le : "*Non mentionné explicitement dans l'expérience.*"
      • **Fais le pont formation→compétence** : "*Cependant, la formation en [module/TP/outil] est étroitement liée à X (raisons 1–2).*"
      • Donne un **niveau de confiance** (Élevé/Moyen/Faible) selon la proximité formation↔X.
      • Propose 1 **question de clarification** ou 3 **étapes d'action**.
   c) **Montée en compétence (si pertinent)** :
      • Mentionne que **les bases acquises en formation** (ex. stats avancées, GLM, processus stochastiques, optimisation, RO/Xpress, EDP/FreeFEM, signal & image sans DL, POO Python/C++) **+ les qualités listées dans “Forces”** (ex. apprentissage rapide, adaptabilité, rigueur) **soutiennent une montée en compétence rapide** sur X.
      • Formulation attendue : "*Non indiqué en expérience. **Appui formation** : [bases précises]. **Qualités** : [2–3 forces]. ⇒ **Montée en compétence rapide** attendue sur X.*"
   d) Si totalement hors champ → refuse poliment et propose de reformuler.
5) Format recommandé :
   - **Verdict** : Oui / Probable (inférence) / Non indiqué
   - **Pourquoi** : 2–4 puces (mapping besoin X ↔ éléments du CV ou **Formation**)
   - **Appui formation** (si utilisé) : 1–2 puces citant explicitement les modules/outils (ex. *RO/Xpress*, *EDP/FreeFEM*, *traitement du signal (convolutions/débruitage)*, *stats avancées (bayésien/GLM/Kalman)*)
   - **Montée en compétence** (si utilisé) : 1 puce rappelant **bases** + **qualités** → montée en compétence rapide
   - **Confiance** : Élevé / Moyen / Faible
   - **Next step** : 1 question de précision OU 3 étapes concrètes

# EXEMPLES DE TON (FR)
- "Verdict : Probable (inférence). Pourquoi : CI/CD + Docker + FastAPI + AWS vus en projet → éléments d'une pipeline MLOps. **Appui formation** : optimisation/ML math + validation croisée. **Montée en compétence** : bases solides + apprentissage rapide ⇒ ramp-up rapide. Confiance : Élevé. Next step : préciser l'outillage de monitoring (logs/metrics/alerts)."
- "Non indiqué dans l'expérience. **Appui formation** : *stats avancées (bayésien/GLM), processus stochastiques, théorie de la mesure* ⇒ bases actuariat. **Montée en compétence** : rigueur + adaptabilité ⇒ ramp-up rapide sur tarification (fréquence/coût). Confiance : Moyen. Next step : outils cibles (SAS/R/Python, stack BI) ?"
"""
    prefix = "Réponds en français professionnel.\n" if style == "pro (FR)" else "Reply in professional English.\n"
    return prefix + base




# UI — Sidebar
st.sidebar.title("⚙️ Options")

model = st.sidebar.selectbox(
    "Modèle",
    [DEFAULT_MODEL, "llama-3.3-8b-instant"],
    index=0,
)
sys_style = st.sidebar.selectbox("Style de réponse", ["pro (FR)", "pro (EN)"], index=0)


#  Init session
def user_count() -> int:
    return sum(1 for m in st.session_state.messages if m["role"] == "user")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": build_system_prompt(CV_TEXT, sys_style)}]
    st.session_state.blocked = False  

# Rebuild prompt si style change
if st.session_state.get("sys_style") != sys_style:
    st.session_state.messages[0] = {"role": "system", "content": build_system_prompt(CV_TEXT, sys_style)}
    st.session_state.sys_style = sys_style

# État blocage dur 
asked = user_count()
if asked >= MAX_QUESTIONS:
    st.session_state.blocked = True


#  En-tête + compteur + verrou

st.title("💼 CV — Jérôme TAM")
st.caption("Streamlit + Groq (stream). Limite stricte à 10 questions par session.")

remaining = max(0, MAX_QUESTIONS - asked)
st.info(f"Questions restantes : **{remaining}/{MAX_QUESTIONS}**")
st.progress(asked / MAX_QUESTIONS if MAX_QUESTIONS else 0.0)

if st.session_state.blocked:
    st.error("Limite atteinte. Le chat est verrouillé pour cette session.")
    disabled_all = True
else:
    disabled_all = False


# Historique 

for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])


# Suggestions (désactivées si blocage)

with st.expander("💡 Suggestions", expanded=True):
    cols = st.columns(3)
    examples = [
        "Peux-tu résumer le profil de Jérôme en 3 phrases ?",
        "Prépare 5 bullets pour 'Parlez-moi de vous'.",
        "Réponses types : forces, faiblesses, prétentions salariales.",
        "Donne 3 exemples STAR (Situation/Tâche/Action/Résultat).",
        "Pitch FR puis traduction EN.",
        "Liste des projets concrets de Jérôme (avec métriques).",
        "Expertise de Jérôme (simplifié)."
    ]
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, use_container_width=True, key=f"sugg_{i}", disabled=disabled_all):
            st.session_state.user_prefill = ex


# Groq streaming

def stream_groq(messages, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for chunk in resp:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content


#  Input (verrouillé si blocage)

placeholder = "Pose une question liée au profil (compétences, expériences, soft skills, MLOps, salaire, dispo)…"
prompt = st.chat_input(placeholder=placeholder, key="chat_input", disabled=disabled_all)

if prompt is None and "user_prefill" in st.session_state and not disabled_all:
    prompt = st.session_state.pop("user_prefill")


#  Gate & exécution 

if prompt:

    if user_count() >= MAX_QUESTIONS or st.session_state.blocked:
        st.session_state.blocked = True
        with st.chat_message("assistant"):
            st.error("Limite atteinte. Le chat est verrouillé pour cette session.")
    else:
      
        if not is_on_topic(prompt):
            with st.chat_message("assistant"):
                st.info("Je réponds uniquement aux questions liées au **profil du candidat** "
                        "(compétences, expériences, soft skills, MLOps, salaire, disponibilité).")
        else:
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                stream_area = st.empty()
                chunks = stream_groq(st.session_state.messages, model)
                full = stream_area.write_stream(chunks)

            st.session_state.messages.append({"role": "assistant", "content": full})

            
            if user_count() >= MAX_QUESTIONS:
                st.session_state.blocked = True
                st.toast("⚠️ Limite atteinte : le chat est désormais verrouillé.")