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




# System prompt 
def build_system_prompt(cv_text: str, style: str) -> str:
    cv_snippet = cv_text[:6000]  # limite tokens
    base = f"""
Tu joues le rôle du candidat dont le CV suit, et tu réponds comme en entretien.
- Langue: si l'interlocuteur parle français → réponds en **je** en français naturel; sinon en anglais avec **I**.
- Ton: bref, chaleureux, professionnel; phrases naturelles; pas de jargon inutile; 3–6 bullets max si utile.
- Quand on te demande "peux-tu faire X ?" :
  1) Si l'information est **clairement présente** dans le CV → réponds **oui**/**non** simplement, puis illustre en 2–4 points concrets tirés du CV (expériences, projets, outils).
  2) Si **ce n'est pas mentionné** dans le CV → dis-le simplement: "D’après mon CV, je n’ai pas encore pratiqué X."
     - Ensuite, fais un **pont crédible**: "En revanche, j’ai [compétence/formation voisine Y] qui s’en rapproche (par ex. ...)."
     - Conclus humainement : "Je peux monter vite en compétence" (+ 1 phrase sur ta démarche: apprendre vite, documenter, demander un cadrage, livrer un POC).
  3) Si la question est trop vague → pose **une seule** question de clarification.
- Interdits: niveaux de confiance, scores inventés, expériences non présentes; pas de phrases type "Verdict", "Confiance", etc.
- Préfère le **premier degré humain**: "Oui, parce que…" / "Pas encore, mais…" / "Voilà comment je m’y prends…"

CV (verbatim) :
---
{cv_snippet}
---

# Gabarits de réponses (FR)

## Compétence clairement présente
"Oui, sur ce point je suis à l’aise.
- [Preuve 1 tirée du CV]
- [Preuve 2 tirée du CV]
- [Outil/techno pertinente]
Si besoin, je peux vous détailler la démarche / un exemple concret."

## Compétence non mentionnée dans le CV (pont formation/bases)
"D’après mon CV, je n’ai pas encore pratiqué X directement.
En revanche, j’ai des bases qui s’en rapprochent :
- [Module/projet/outil voisin 1]
- [Module/projet/outil voisin 2]
Je peux monter vite en compétence : je commence par un petit cadrage, un POC mesurable, puis j’industrialise si ça valide."

## Question trop vague
"Pour être précis, vous pensez à quel contexte pour X ? (ex : outil attendu, volume de données, délai)"
"""
    prefix = "Réponds en français, à la première personne (je).\n" if style == "pro (FR)" else "Answer in English, first person (I).\n"
    return prefix + base





# UI — Sidebar
st.sidebar.title("⚙️ Options")

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
st.caption("Limite stricte à 10 questions par session.")

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
    ]
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, use_container_width=True, key=f"sugg_{i}", disabled=disabled_all):
            st.session_state.user_prefill = ex


# Groq streaming

def stream_groq(messages, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.4,   
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
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_area = st.empty()
            chunks = stream_groq(st.session_state.messages, DEFAULT_MODEL)
            full = stream_area.write_stream(chunks)

        st.session_state.messages.append({"role": "assistant", "content": full})

            
        if user_count() >= MAX_QUESTIONS:
            st.session_state.blocked = True
            st.toast("⚠️ Limite atteinte : le chat est désormais verrouillé.")