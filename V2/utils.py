"""
utils.py
========
Módulo de utilidades transversales del sistema: detección de emociones,
detección de riesgo de crisis (depresión), generación del resumen
acumulado entre sesiones, y los diccionarios de presentación (emojis y
traducción de etiquetas de emoción al español) usados por la interfaz.

Estas funciones no orquestan agentes ni definen prompts; son los "métodos"
de soporte (modelos de Hugging Face, cálculos y mapeos) reutilizados tanto
por `chat.py` (interfaz) como indirectamente por `agents.py`.
"""

import json
import warnings

import torch
import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from crisis_patterns import detectar_riesgo_regex
import database as db
from agents import get_llm

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

@st.cache_resource


def load_emotion_detector():
    tokenizer = AutoTokenizer.from_pretrained("AnasAlokla/multilingual_go_emotions")
    model = AutoModelForSequenceClassification.from_pretrained("AnasAlokla/multilingual_go_emotions")
    model.eval()
    return tokenizer, model

emotion_tokenizer, emotion_model = load_emotion_detector()
EMOTION_LABELS = [emotion_model.config.id2label[i] for i in range(len(emotion_model.config.id2label))]

def emotion_detector(query: str) -> str:
    inputs = emotion_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=192)
    with torch.no_grad():

        probs = torch.sigmoid(emotion_model(**inputs).logits).cpu().numpy()[0]

    all_emotions = [{"label": EMOTION_LABELS[i], "score": float(probs[i])} for i in range(len(EMOTION_LABELS))]
    all_emotions.sort(key=lambda x: x["score"], reverse=True)
    
    result = [e for e in all_emotions if e["score"] >= 0.05]
    
        
    return json.dumps(result)


@st.cache_resource
def load_translator():

   return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

@st.cache_resource
def load_depression_detector():
    return pipeline("text-classification", model="rafalposwiata/deproberta-large-depression")

depression_classifier = load_depression_detector()
translator = load_translator()

def build_cumulative_summary(user_id: str) -> str:
    sessions = db.get_all_sessions()
    sesiones_usuario = [s for s in sessions if s["user_id"] == user_id]

    fragmentos = []
    vistos = set()


    for s in sesiones_usuario:
        resumen = db.get_conversation_summary(s["id"])
        if resumen:
            for frag in resumen.split("|"):
                frag = frag.strip()
                if frag and frag.lower() not in vistos:
                    fragmentos.append(frag)
                    vistos.add(frag.lower())

    for s in sesiones_usuario:
        eventos = db.get_events(s["id"])
        for e in eventos:
            frag = f"Evento: {e['event']}" + (f" ({e['type']})" if e['type'] else "")
            if frag.lower() not in vistos:
                fragmentos.append(frag)
                vistos.add(frag.lower())

    perfil_texto = db.get_user_profile_text(user_id)
    if perfil_texto and "No hay perfil" not in perfil_texto:
        fragmentos.append(f"Perfil: {perfil_texto}")

    if not fragmentos:
        return ""

    texto_bruto = " | ".join(fragmentos[-20:])

    try:
        llm = get_llm(reasoning=False)
        prompt_condensar = (
            "Resume en un único párrafo breve y coherente (máximo 4-5 frases) "
            "los siguientes hechos sobre un usuario, eliminando repeticiones y "
            "conservando solo la información relevante:\n\n"
            f"{texto_bruto}"
        )
        respuesta = llm.invoke(prompt_condensar)
        return respuesta.content.strip()
    except Exception as e:
        print(f"[WARN] No se pudo condensar el resumen acumulado: {e}")
        return texto_bruto


EMOCIONES_NEGATIVAS = {
    "anger", "annoyance", "disapproval", "disgust", "embarrassment",
    "fear", "grief", "nervousness", "remorse", "sadness", "disappointment","desire",
}

EMOCIONES_POSITIVAS = {
    "admiration", "approval", "caring", "curiosity",
    "excitement", "gratitude", "joy", "love",
    "optimism", "pride", "realization", "relief", "surprise"
}


def tiene_mas_emociones_negativas(emotion_data: list, umbral: float = 0.30) -> list:
    neg = 0
    neg_flag = False
    pos = 0
    if not emotion_data:
        return False
    for emo in emotion_data:
        if emo["label"] in EMOCIONES_NEGATIVAS:
            neg += 1
            if emo["score"] >= umbral:
                neg_flag = True
        elif emo["label"] in EMOCIONES_POSITIVAS:
            pos += 1
    return [neg > pos, neg_flag]



def evaluar_riesgo_crisis(user_input: str, emotion_data: list) -> bool:
    if not user_input.strip():
        return False

    # Red de seguridad: si hay match crítico, no dependemos del modelo
    regex_result = detectar_riesgo_regex(user_input)
    if regex_result["match_critico"]:
        print(f"Crisis detectada por regex: {regex_result['motivo']}")
        return True

    try:
        traduccion = translator(user_input)[0]['translation_text']
        resultado = depression_classifier(traduccion)[0]
        print(f"Traducción: {traduccion}")
        print(f"Resultado: {resultado}")
        if resultado['label'] == 'moderate':
            if tiene_mas_emociones_negativas(emotion_data, umbral=0.05)[0] or tiene_mas_emociones_negativas(emotion_data, umbral=0.05)[1]:
                print("Se detecta riesgo de crisis: emociones negativas predominantes o confianza alta.")
                return True
            else:
                return False
        elif resultado['label'] == 'severe':
            return True
        else:
            return False

    except Exception as e:
        print(f"Error en Capa 2: {e}")
        return True


EMOJI_MAP = {
    "admiration": "🤩",
    "amusement": "😄",
    "anger": "😠",
    "annoyance": "😒",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "😕",
    "curiosity": "🧐",
    "desire": "😍",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤢",
    "embarrassment": "😳",
    "excitement": "🤩",
    "fear": "😨",
    "gratitude": "🙏",
    "grief": "💔",
    "joy": "😊",
    "love": "❤️",
    "nervousness": "😬",
    "neutral": "😐",
    "optimism": "🌤️",
    "pride": "🦁",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔",
    "sadness": "😢",
    "surprise": "😲",
}
 
EN_TO_ES_MAP = {
    "admiration": "Admiración",
    "amusement": "Diversión",
    "anger": "Rabia",
    "annoyance": "Fastidio",
    "approval": "Aprobación",
    "caring": "Cariño",
    "confusion": "Confusión",
    "curiosity": "Curiosidad",
    "desire": "Deseo",
    "disappointment": "Decepción",
    "disapproval": "Desaprobación",
    "disgust": "Asco",
    "embarrassment": "Vergüenza",
    "excitement": "Entusiasmo",
    "fear": "Miedo",
    "gratitude": "Gratitud",
    "grief": "Duelo",
    "joy": "Alegría",
    "love": "Amor",
    "nervousness": "Nerviosismo",
    "neutral": "Neutral",
    "optimism": "Optimismo",
    "pride": "Orgullo",
    "realization": "Comprensión",
    "relief": "Alivio",
    "remorse": "Remordimiento",
    "sadness": "Tristeza",
    "surprise": "Sorpresa",
}
