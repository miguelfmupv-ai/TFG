"""
chat.py
=======
Capa de interfaz (Streamlit) del sistema de apoyo emocional conversacional.

Este módulo se encarga exclusivamente de:
    - Gestionar el estado de sesión de Streamlit (toggles, sesión activa, etc.).
    - Renderizar la interfaz (chat, barra lateral, diálogos de gestión).
    - Orquestar el flujo de un turno de conversación: detección de emociones,
      evaluación de riesgo de crisis, invocación del agente y persistencia
      de mensajes en la base de datos.

Toda la lógica de prompts vive en `prompts.py`, la orquestación de agentes
LangChain en `agents.py`, y las utilidades de emociones/crisis en `utils.py`.
"""

import streamlit as st
import os
import database as db
import asyncio
import nest_asyncio
import sys
import json

from agents import run_agent, run_reset_agent
from utils import (
    emotion_detector,
    evaluar_riesgo_crisis,
    build_cumulative_summary,
    EMOJI_MAP,
    EN_TO_ES_MAP,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

nest_asyncio.apply()

st.set_page_config(page_title="Validador emocional", page_icon="🧠")
st.title("🧠 Validador Emocional", anchor = False)


if "user_id" not in st.session_state or "session_id" not in st.session_state:

    st.session_state.user_id = db.get_or_create_profile()
    

    sessions = db.get_all_sessions() 
    if sessions:
        most_recent = sessions[0]
        st.session_state.session_id = most_recent["id"]
    else:
        st.session_state.session_id = db.create_session("Conversación actual", user_id=st.session_state.user_id)


if "emotion_detection_enabled" not in st.session_state:
    st.session_state.emotion_detection_enabled = True
    
if "memory_enabled" not in st.session_state:
    st.session_state.memory_enabled = True

if "crisis_detected" not in st.session_state:
    st.session_state.crisis_detected = False

if "summarizer_enabled" not in st.session_state:
    st.session_state.summarizer_enabled = True

if "emotion_prediction_enabled" not in st.session_state:
    st.session_state.emotion_prediction_enabled = True

if "reasoning_enabled" not in st.session_state:
    st.session_state.reasoning_enabled = True


if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


if st.session_state.crisis_detected or db.get_crisis_status(st.session_state.user_id):
    st.session_state.crisis_detected = True
    st.error("🚨 **ALERTA MÁXIMA DE CRISIS** 🚨\n\n"
            "Por favor, detente y busca ayuda de inmediato. Tu vida es increíblemente valiosa y mereces sentirte mejor.\n\n"
            "Si estás en peligro inminente, por favor, llama inmediatamente a los servicios de emergencia. "
            "Recuerda que no tienes que pasar por esto solo/a. Hay profesionales que están entrenados y listos para ayudarte en este momento. "
            "Por favor, utiliza estos recursos de ayuda profesional:\n\n"
            "* 📞 **Línea de Prevención del Suicidio:** Llama gratis al **024** (Atención a la conducta suicida).\n"
            "* 🚑 **Servicios de Emergencia:** Llama al **112**.\n"
            "* 🫂 **Busca a alguien de confianza:** Llama a un amigo, familiar o persona cercana y dile que necesitas ayuda ahora mismo.\n\n"
            "Por favor, haz una pausa. Respira hondo. Yo estoy aquí para escucharte y seguirte acompañando, pero en este momento, "
            "lo más importante es que hables con un profesional. Mándame un mensaje cuando estés en un lugar seguro y con ayuda profesional. "
            "Estamos juntos en esto.")
    st.chat_input("Chat desactivado", disabled=True)
    st.stop()


@st.dialog("Gestionar perfil")
def profile_dialog():
    st.caption(
        "Dime qué quieres añadir, cambiar o eliminar de algún campo específico.\n\n"
        "**Ejemplos:**\n"
        "- Añadir: *\"añade como afición que me gusta la fotografía\"*\n"
        "- Modificar: *\"cambia mi tema recurrente 'Amigos' por 'Ruptura'\"*\n"
        "- Eliminar: *\"borra mi objetivo de aprender inglés\"*\n"
        "- Borrar todo un campo: *\"borra todos mis temas\"*"
    )
    user_input = st.chat_input("Ej: cambia mi tema recurrente 'Amigos' por 'Ruptura'")
    if user_input:
        with st.spinner("Procesando..."):
            response = asyncio.run(run_reset_agent(user_input, mode="perfil"))
            for msg in st.session_state.pop("pending_toasts", []):
                placeholder = st.empty()
                placeholder.success(msg)
        st.markdown(response)
        if st.button("✅ Aceptar", type="primary", use_container_width=True):
            st.rerun()


@st.dialog("Gestionar eventos")
def events_dialog():
    st.caption(
        "Dime qué evento quieres añadir, cambiar o eliminar.\n\n"
        "**Ejemplos:**\n"
        "- Añadir: *\"añade que me caí en el trabajo\"*\n"
        "- Modificar: *\"cambia la importancia del evento del ascenso a media\"*\n"
        "- Eliminar: *\"borra el evento del despido\"*\n"
        "- Borrar todos: *\"borra todos los eventos de esta sesión\"*"
    )
    user_input = st.chat_input("Ej: cambia la importancia del evento del ascenso a media")
    if user_input:
        with st.spinner("Procesando..."):
            response = asyncio.run(run_reset_agent(user_input, mode="eventos")) 
            for msg in st.session_state.pop("pending_toasts", []):
                placeholder = st.empty()
                placeholder.success(msg)
        st.markdown(response)
        if st.button("✅ Aceptar", type="primary", use_container_width=True):
            st.rerun()



@st.dialog("Gestionar resumen")
def summary_dialog():
    st.caption(
        "Dime qué quieres añadir, cambiar o eliminar del resumen de esta sesión de forma concreta.\n\n"
        "**Ejemplos:**\n"
        "- Añadir: *\"añade que debido a mi situación laboral estoy muy estresado\"*\n"
        "- Modificar: *\"cambia la parte del ascenso por 'ascenso a gerente'\"*\n"
        "- Eliminar: *\"elimina la parte sobre el accidente laboral\"*\n"
        "- Borrar todo: *\"borra todo el resumen\"*"
    )
    user_input = st.chat_input("Ej: elimina la parte sobre el accidente laboral")
    if user_input:
        with st.spinner("Procesando..."):
            response = asyncio.run(run_reset_agent(user_input, mode="resumen"))
            for msg in st.session_state.pop("pending_toasts", []):
                placeholder = st.empty()
                placeholder.success(msg)
        st.markdown(response)
        if st.button("✅ Aceptar", type="primary", use_container_width=True):
            st.rerun()

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-top: -4rem; padding: 0 0 0.5rem 0;">
            <span style="font-size: 4rem;">🧠</span>
            <h1 style="font-size: 2rem; font-weight: 700; margin: 0.0rem 0 0.2rem 0;">
                Validador Emocional
            </h1>
            <p style="font-size: 1rem; opacity: 0.6; margin: 0;">
                Tu espacio seguro para hablar
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
    if st.button("✏️ Nueva conversación", type="primary", use_container_width=True):
        cumulative = build_cumulative_summary(st.session_state.user_id)
        new_id = db.create_session("Nueva conversación", user_id=st.session_state.user_id)
        if cumulative:
            db.update_session(new_id, new_summary=cumulative)
        st.session_state.session_id = new_id
        st.rerun()
    st.divider()
    st.session_state.emotion_detection_enabled = st.toggle(
        "🧠 Detector de emociones",
        value=st.session_state.emotion_detection_enabled,
        help="Activa o desactiva el análisis emocional de tus mensajes."
    )

    st.toggle(
    "🔮 Predicción emocional",
    key="emotion_prediction_enabled",
    disabled=not st.session_state.emotion_detection_enabled,
    help="Predice el estado emocional del usuario basándose en sesiones anteriores."
    )

    st.divider()

    def on_memory_change():
        if not st.session_state.memory_enabled:
            st.session_state.summarizer_enabled = False


    st.toggle(
        "🗃️ Memoria",
        key="memory_enabled",
        on_change=on_memory_change,
        help="Activa o desactiva el guardado y recuperación de información sobre ti."
    )
    st.toggle(
        "📝 Resumen de sesión",
        key="summarizer_enabled",
        disabled=not st.session_state.memory_enabled,
        help="Cuando está activo, el agente guarda un resumen al detectar eventos importantes."
    )

    st.toggle(
    "🧩 Razonamiento extendido",
    key="reasoning_enabled",
    help="Para mensajes largos o complejos. No aconsejado para mensajes cortos ni conversaciones rápidas, ya que puede ralentizar la respuesta del agente."
    )

    st.divider()
    st.markdown("**👤 Lo que sé de ti**")
    profile_text = db.get_user_profile_text(st.session_state.user_id)
    

    profile_text = profile_text.replace(".", ". ").replace("  ", " ")
    
    VALORES_VACIOS = {"No indicado", "No indicadas", "No indicados", "Ninguno", "Ninguna", ""}

    if "No hay perfil" in profile_text:
        st.caption("_Todavía no tengo información tuya._")
    else:
        for part in profile_text.split(". "):
            part = part.strip()
            if ": " in part:
                label, value = part.split(": ", 1)
                label = label.strip()
                value = value.strip()
                st.caption(f"**{label}:** {value}")
  
    
    if st.button("✏️ Gestionar perfil", use_container_width=True):
        profile_dialog()
    
    st.divider()

    st.write("🟢 Sesión actual:")
    all_sessions = db.get_all_sessions()
    current_session = next((s for s in all_sessions if s["id"] == st.session_state.session_id), None)
    other_sessions  = [s for s in all_sessions if s["id"] != st.session_state.session_id]

    if current_session:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.button("▶ " + (current_session["name"] or "Sin nombre"),
                    key=current_session["id"], type="primary",
                    use_container_width=True, disabled=True)
        with col2:
            if st.button("🗑", key=f"del_{current_session['id']}"):
                db.delete_session(current_session["id"])
                if other_sessions:
                    st.session_state.session_id = other_sessions[0]["id"]
                else:
                    st.session_state.session_id = db.create_session("Nueva conversación", user_id=st.session_state.user_id)
                st.rerun()

    st.markdown("**📝 Resumen de sesión**")
    summary = db.get_conversation_summary(st.session_state.session_id)
    if summary:
        st.caption(summary)
    else:
        st.caption("_Todavía no hay resumen para esta sesión._")

    if st.button("✏️ Gestionar resumen", use_container_width=True):
        summary_dialog()

    st.divider()
    st.markdown("**📌 Eventos de la sesión**")
    events = db.get_events(st.session_state.session_id)
    if events:
        for event in events:
            with st.container(border=True):
                st.caption(f"**Evento:** {event['event']}")
                st.caption(f"**Tipo:** {event['type'] or 'No especificado'}")
                st.caption(f"**Importancia:** {event['importance'] or 'No especificada'}")
    else:
        st.caption("_No hay eventos registrados en esta sesión._")

    if st.button("✏️ Gestionar eventos", use_container_width=True):
        events_dialog()

    st.divider()
    st.write("📁 Sesiones:")
    for session in other_sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(session["name"] or "Sin nombre", key=session["id"], use_container_width=True):
                st.session_state.session_id = session["id"]
                st.session_state.user_id = session["user_id"]
                st.session_state.crisis_detected = False
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{session['id']}"):
                db.delete_session(session["id"])
                st.rerun()

    st.divider()
    if st.button("🔄 Recargar la sesión", type="primary", use_container_width=True):
        st.rerun()
    st.divider()
    if "confirm_exit" not in st.session_state:
        st.session_state.confirm_exit = False

    if not st.session_state.confirm_exit:
        if st.button("🔴 Salir y cerrar aplicación", use_container_width=True):
            st.session_state.confirm_exit = True
            st.rerun()
    else:
        st.warning("¿Seguro que quieres cerrar la aplicación?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Sí, cerrar", use_container_width=True):
                os._exit(0)
        with col2:
            if st.button("❌ Cancelar", use_container_width=True):
                st.session_state.confirm_exit = False
                st.rerun()




for message in db.get_messages(st.session_state.session_id):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if message["role"] == "user" and message.get("detected_emotion"):
        try:
            emotion_data = json.loads(message["detected_emotion"])
            if emotion_data:
                top = sorted(emotion_data, key=lambda x: x['score'], reverse=True)[:5]
                cols = st.columns(len(top))
                for i, emo in enumerate(top):
                    with cols[i]:
                        label_en = emo['label']
                        emoji = EMOJI_MAP.get(label_en, '🔹')
                        label_es = EN_TO_ES_MAP.get(label_en, label_en.capitalize())
                        st.caption(f"{emoji} {label_es} {emo['score']:.0%}")
        except (json.JSONDecodeError, TypeError):
            pass



input_value = st.chat_input("", disabled=(st.session_state.crisis_detected or st.session_state.is_thinking))

if input_value:
    st.session_state.pending_input = input_value
    st.session_state.is_thinking = True
    st.rerun()


if st.session_state.pending_input:
    user_input = st.session_state.pending_input

    with st.chat_message("user"):
        st.markdown(user_input)


    emotion_data = []
    predicted_emotion_dist = {}
    predicted_emotion = None

    if st.session_state.emotion_detection_enabled:
        emotions = emotion_detector(user_input)
    else:
        emotions = json.dumps([])
        
    db.save_message(st.session_state.session_id, "user", user_input, emotions)
    
    mensajes = db.get_messages(st.session_state.session_id)
    if len(mensajes) == 1:

        if len(user_input) > 40:

            nuevo_nombre = user_input[:40].rsplit(' ', 1)[0] + "..."
        else:
            nuevo_nombre = user_input
            
        db.update_session(st.session_state.session_id, new_name=nuevo_nombre)


    if st.session_state.emotion_detection_enabled:
        emotion_data = json.loads(emotions)
        predicted_emotion_dist = (
            db.predict_next_emotion_distribution(st.session_state.session_id)
            if st.session_state.emotion_prediction_enabled
            else {}
        )
        predicted_emotion = max(predicted_emotion_dist, key=predicted_emotion_dist.get) if predicted_emotion_dist else None
        print(f"Predicción: {predicted_emotion}")
        db.log_session_emotions(st.session_state.session_id, emotion_data)
        scores = {d['label']: d['score'] for d in emotion_data}
        top = sorted(emotion_data, key=lambda x: x['score'], reverse=True)[:5]
        st.info("🔍 Emociones detectadas\n\nℹ️ *Los porcentajes indican la confianza de que esa emoción esté presente en el mensaje. Al ser independientes, la suma puede superar el 100%.*")
        cols = st.columns(len(top))
        for i, emo in enumerate(top):
                with cols[i]:
                    label_en = emo['label']
                    emoji = EMOJI_MAP.get(label_en, '🔹')
                    label_es = EN_TO_ES_MAP.get(label_en, label_en.capitalize())
                    
                    st.metric(
                        label=f"{emoji} {label_es}",
                        value=f"{emo['score']:.0%}"
                    )
    
    if st.session_state.emotion_prediction_enabled and predicted_emotion_dist:
        top_pred = sorted(predicted_emotion_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        texto_pred = ", ".join([
            f"{EN_TO_ES_MAP.get(label, label)} {EMOJI_MAP.get(label, '')} ({score:.0%})"
            for label, score in top_pred
        ])
        st.caption(f"🔮 Emoción predicha para esta sesión: {texto_pred}")

    if st.session_state.emotion_detection_enabled and evaluar_riesgo_crisis(user_input, emotion_data):
        st.session_state.crisis_detected = True
        st.error("🚨 **ALERTA MÁXIMA DE CRISIS** 🚨\n\n"
                "Por favor, detente y busca ayuda de inmediato. Tu vida es increíblemente valiosa y mereces sentirte mejor.\n\n"
                "Si estás en peligro inminente, por favor, llama inmediatamente a los servicios de emergencia. "
                "Recuerda que no tienes que pasar por esto solo/a. Hay profesionales que están entrenados y listos para ayudarte en este momento. "
                "Por favor, utiliza estos recursos de ayuda profesional:\n\n"
                "* 📞 **Línea de Prevención del Suicidio:** Llama gratis al **024** (Atención a la conducta suicida).\n"
                "* 🚑 **Servicios de Emergencia:** Llama al **112**.\n"
                "* 🫂 **Busca a alguien de confianza:** Llama a un amigo, familiar o persona cercana y dile que necesitas ayuda ahora mismo.\n\n"
                "Por favor, haz una pausa. Respira hondo. Yo estoy aquí para escucharte y seguirte acompañando, pero en este momento, "
                "lo más importante es que hables con un profesional. Mándame un mensaje cuando estés en un lugar seguro y con ayuda profesional. "
                "Estamos juntos en esto.")
        db.save_message(st.session_state.session_id, "assistant", "ALERTA DE CRISIS: El modelo ha detectado un riesgo crítico de depresión/crisis en el mensaje del usuario.")
        

        st.session_state.pending_input = None
        st.session_state.is_thinking = False
        st.rerun()
       
    else:
        
        with st.chat_message("assistant"):
            with st.spinner("Escribiendo..."):
                try:
                    response = asyncio.run(asyncio.wait_for(
                        run_agent(
                            user_input, emotions,
                            st.session_state.memory_enabled,
                            st.session_state.summarizer_enabled,
                            st.session_state.emotion_detection_enabled,
                            predicted_emotion,
                            st.session_state.emotion_prediction_enabled
                        ),
                        timeout=60.0
                    ))

                    if not response or not response.strip():
                        raise RuntimeError("Respuesta vacía del agente")
                    if "Agent stopped" in response:
                        raise RuntimeError("El agente no completó el razonamiento (límite de iteraciones)")

                except asyncio.TimeoutError:
                    print("[ERROR] Timeout: el agente tardó más de 60s")
                    response = "Estoy tardando más de lo normal en responder. ¿Puedes reformular tu mensaje o intentarlo de nuevo?"

                except ConnectionError:
                    print("[ERROR] Fallo de conexión con Ollama")
                    response = "Parece que no puedo conectar con el modelo en este momento. Comprueba que Ollama esté activo e inténtalo de nuevo."

                except RuntimeError as e:
                    print(f"[ERROR] {e}")
                    response = "He tenido un problema procesando tu mensaje. ¿Puedes intentarlo de nuevo con otras palabras?"

                except Exception as e:
                    print(f"[ERROR] Error inesperado: {type(e).__name__}: {e}")
                    response = "Ha ocurrido un error inesperado. Si el problema persiste, prueba a recargar la sesión."

                st.markdown(response)
                
        db.save_message(st.session_state.session_id, "assistant", response)
        

        st.session_state.pending_input = None
        st.session_state.is_thinking = False
        st.rerun()