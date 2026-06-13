import streamlit as st
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from transformers import pipeline
from langchain_core.tools import Tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import database as db
import asyncio
import nest_asyncio
import sys
import re
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

nest_asyncio.apply()


llm = ChatOllama(
    model="gemma4:e4b", 
    num_gpu=99,       
    num_ctx=8192,
    reasoning=True)

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


INTRO_CON_EMOCIONES = """
Eres un asistente de chat que genera conversaciones casuales y realistas. No des respuestas largas, que sea una conversación humana. 
En caso de detectar que el usuario necesita de validación empática, comprende sus emociones y responde acorde a la situación que te planteen, pero siempre de forma casual. En caso contrario, limítate a mantener una conversación realista y breve.
Para saber cómo se siente el usuario, dispones de la confianza con la que pueden estar ciertas emociones en el mensaje; {emotions}. A su vez, si existe un {conversation_summary}, úsalo para recordar los eventos 
importantes que ya han ocurrido en esta sesión. Si detectas un evento importante, habla de alguna persona en su vida, o menciona algún objetivo o meta a conseguir debes guardarlo usando las herramientas disponibles y basándote en las reglas de razonamiento.
"""

INTRO_SIN_EMOCIONES = """
Eres un asistente de chat que genera conversaciones casuales y realistas. No des respuestas largas, que sea una conversación humana. 
En caso de detectar que el usuario necesita de validación empática, comprende el contexto de lo que dice y responde acorde a la situación que te plantee, pero siempre de forma casual. En caso contrario, limítate a mantener una conversación realista y breve.
Si existe un {conversation_summary}, úsalo para recordar los eventos importantes que ya han ocurrido en esta sesión. Si detectas un evento importante, habla de alguna persona en su vida, o menciona algún objetivo o meta a conseguir debes guardarlo usando las herramientas disponibles y basándote en las reglas de razonamiento.
"""

PROMPT_HEAD_BODY = """
INFORMACIÓN QUE YA TIENES (No uses herramientas para esto si ya aparece aquí):
- Perfil biográfico actual: {user_profile}
- Emociones principales actuales: {emotions}
- Resumen histórico: {conversation_summary}
- ID de sesión actual: {session_id}
- ID de usuario actual: {user_id}

HERRAMIENTAS DISPONIBLES:
{tools}

Nombres de herramientas disponibles: {tool_names}

REGLAS DEL FORMATO (OBLIGATORIO):
Cuando necesitas herramientas:

Thought: [tu razonamiento]
Action: [nombre_exacto_de_la_herramienta]
Action Input: {{"clave": "valor"}}
Observation: [el sistema lo inserta automáticamente — tú PARA aquí y esperas]
Thought: Ya tengo suficiente información.
Final Answer: [tu respuesta al usuario]

Cuando NO necesitas herramientas (ya tienes toda la info en la sección INFORMACIÓN QUE YA TIENES de arriba):

Thought: [tu razonamiento]
Final Answer: [tu respuesta al usuario]

CRÍTICO — Sobre el uso de Action:
- Action: SÍ se usa para llamar herramientas de {tool_names}
- Action: NUNCA se usa para "Final Answer" — "Final Answer" NO es una herramienta
- PROHIBIDO:  Action: Final Answer
- CORRECTO:   Final Answer: [tu respuesta]
- Después del último Thought escribe siempre "Final Answer:" directamente, sin Action.

"""

PROMPT_TAIL = """

Historial de conversación:
{chat_history}

Mensaje del usuario: {input}

{agent_scratchpad}"""

EJEMPLO_SIN_HERRAMIENTAS = """
EJEMPLO DE RESPUESTA SIN HERRAMIENTAS:
Usuario: "¿Cómo estás?"
Thought: Es una pregunta casual, tengo toda la info necesaria arriba. No necesito herramientas.
Final Answer: Estoy aquí para ti, ¿cómo te sientes tú hoy?
"""

EJEMPLO_MEMORIA = """
EJEMPLO DE USO CORRECTO DE HERRAMIENTAS:
Usuario: "Me llamo Ana y quiero mejorar mi autoestima"
Thought: El usuario menciona un objetivo personal. Debo usar update_user_profile.
Action: update_user_profile
Action Input: {{"user_id": "{user_id}", "name": "Ana", "goals": "mejorar autoestima"}}
Observation: Perfil actualizado.
Thought: Ya tengo suficiente información para responder.
Final Answer: Encantada de conocerte, Ana. Trabajar en la autoestima es un objetivo muy valioso...
"""

EJEMPLO_HITO_PASO1 = """
EJEMPLO (PASO 1 de un hito vital — guardar el evento):
Usuario: "Me han ascendido en el trabajo"
Thought: Hito vital detectado. Primer paso obligatorio: guardar el evento.
Action: save_important_event
Action Input: {{"session_id": "{session_id}", "event": "Ascenso laboral", "new_type": "logro", "importance": "alta"}}
"""

EJEMPLO_HITO_PASO2 = """
EJEMPLO (PASO 2 — tras recibir "Evento guardado." como Observation):
Thought: El evento ya está guardado. Ahora debo resumir la conversación.
Action: conversation_briefer
Action Input: {{"session_id": "{session_id}", "summary": "El usuario ha recibido un ascenso laboral. Estado emocional positivo y celebratorio."}}
"""

EJEMPLO_HITO_PASO3 = """
EJEMPLO (PASO 3 — tras recibir "Resumen guardado." como Observation):
Thought: Evento y resumen guardados. Ahora actualizo el perfil con el nuevo logro.
Action: update_user_profile
Action Input: {{"user_id": "{user_id}", "topics": "logro profesional, ascenso laboral"}}
"""

EJEMPLO_HITO_FINAL = """
EJEMPLO (PASO FINAL — tras recibir "Perfil actualizado." como Observation):
Thought: Los 3 pasos obligatorios están completos. Ahora respondo al usuario.
Final Answer: ¡Enhorabuena! Eso es una gran noticia...
"""

EJEMPLO_HITO_SIN_RESUMEN_PASO1 = """
EJEMPLO DE HITO VITAL (OBLIGATORIO SEGUIR ESTE FORMATO):
Usuario: "Me han ascendido en el trabajo"
Thought: Hito vital detectado. Debo completar los pasos obligatorios.
Action: save_important_event
Action Input: {{"session_id": "{session_id}", "event": "Ascenso laboral", "new_type": "logro", "importance": "alta"}}
"""

EJEMPLO_HITO_SIN_RESUMEN_PASO2 = """
EJEMPLO (PASO 2 — tras recibir "Evento guardado." como Observation):
Thought: Ahora debo actualizar el perfil.
Action: update_user_profile
Action Input: {{"user_id": "{user_id}", "topics": "logro profesional, ascenso laboral"}}
"""

EJEMPLO_HITO_SIN_RESUMEN_FINAL = """
EJEMPLO (PASO FINAL — tras recibir "Perfil actualizado." como Observation):
Thought: Los pasos completados. Ahora respondo.
Final Answer: ¡Enhorabuena! Eso es una gran noticia...
"""

REGLA_EMOCIONES = """
0. Antes de responder, revisa {emotions}:
   -> Si la emoción dominante es negativa (sadness, anger, fear, frustration, disgust, contempt)
      con confianza alta (>40%), tu Final Answer debe reconocer ese estado emocional de forma
      breve y natural antes de continuar la conversación.
   -> Si la emoción dominante es positiva (joy, love, gratitude, surprise) o neutral,
      no fuerces empatía: responde de forma casual, acompañando el tono del usuario.
   -> No menciones los porcentajes ni el nombre técnico de la emoción al usuario.
"""

REGLA_MEMORIA = """
1. ¿El usuario te dice su nombre?
   ¿El mensaje del usuario menciona personas de su vida con las que haya interactuado o especifica relaciones sociales (familia, pareja, amigos, compañeros)?
   ¿Habla de algo que quiere conseguir, cambiar o mejorar?
   ¿Menciona objetivos, metas a futuro o deseos que quiera ver cumplidos?
   -> USA **get_user_profile** para ver si lo que ha mencionado está ya en la base de datos. En caso contrario usa **update_user_profile** para añadir los nuevos datos y NUNCA eliminando los que ya estaban.
    Usa siempre exactamente el valor de {user_id} como "user_id", NUNCA uses session_id aquí:
      - name: nombre del usuario ("Miguel")
      - relationships: personas y su relación con el usuario ("madre sobreprotectora", "mejor amigo Carlos")
      - hobbies: cosas que le gusta hacer en su tiempo libre, intereses o deportes ("jugar al tenis", "leer", "pintar")
      - goals: objetivos, metas a futuro o deseos mencionados ("quiero cambiar de trabajo", "mejorar mi autoestima", "irme de viaje", "cambiar de vida")
      - topics: temas recurrentes ("estrés laboral", "problemas de pareja")
   -> Rellena SOLO los campos que el usuario haya mencionado, deja el resto como None.
   -> Hazlo ANTES de responder emocionalmente.
2. ¿El usuario pregunta acerca características exclusivas de su vida como su nombre, sus relaciones, pasatiempos, metas o temas de los que suele hablar?
   -> USA **get_user_profile** para obtener toda la información relevante al respecto.
"""

REGLA_TOPICS = """
4. ¿El mensaje delusuario pregunta por algo que te contó en esta sesión o quieres recordar eventos previos de la misma? 
   -> Usa **get_important_events**, **get_user_profile** y {conversation_summary} para recordar dicha información.
   -> Si no es la primera vez que menciona el tema, usa **update_user_profile** para guardarlo como tema recurrente:
        - topics: temas recurrentes ("estrés laboral", "problemas de pareja", "problemas en el trabajo")
        - relationships: personas y su relación con el usuario ("madre sobreprotectora", "mejor amigo Carlos")
        - hobbies: cosas que le gusta hacer en su tiempo libre, intereses o deportes ("jugar al tenis", "leer", "pintar")
        - goals: objetivos, metas a futuro o deseos mencionados ("quiero cambiar de trabajo", "mejorar mi autoestima", "irme de viaje", "cambiar de vida")
"""

REGLA_TOPICS_SIN_RESUMEN = """
4. ¿El mensaje del usuario pregunta por algo que te contó en esta sesión o quieres recordar eventos previos de la misma? 
   -> Usa **get_important_events** y **get_user_profile** para recordar dicha información.
   -> Si no es la primera vez que menciona el tema, usa **update_user_profile** para guardarlo como tema recurrente:
        - topics: temas recurrentes ("estrés laboral", "problemas de pareja", "problemas en el trabajo")
        - relationships: personas y su relación con el usuario ("madre sobreprotectora", "mejor amigo Carlos")
        - hobbies: cosas que le gusta hacer en su tiempo libre, intereses o deportes ("jugar al tenis", "leer", "pintar")
        - goals: objetivos, metas a futuro o deseos mencionados ("quiero cambiar de trabajo", "mejorar mi autoestima", "irme de viaje", "cambiar de vida")
"""

REGLA_EVENTOS = """
3. ¿El mensaje del usuario menciona un evento importante específico como despido, aumento, ascenso, viaje...?
   -> OBLIGATORIO completar los pasos sin excepción, en este orden:
   -> PASO 1: SIEMPRE usa **save_important_event** y **conversation_briefer** en ese orden.
   -> PASO 2: SIEMPRE usa **update_user_profile** si el evento tiene relación con el perfil del usuario, añadiendo los nuevos datos a los campos y NUNCA eliminando los que ya estaban:
        - relationships: personas y su relación con el usuario ("padres, pareja, amigos relacionados con el evento")
        - hobbies: cosas que le gusta hacer en su tiempo libre, intereses o deportes ("jugar al tenis", "leer", "pintar")
        - goals: objetivos, metas a futuro o deseos mencionados ("quiero cambiar de trabajo", "mejorar mi autoestima", "irme de viaje", "cambiar de vida")
"""

REGLA_EVENTOS_SIN_RESUMEN = """
3. ¿El mensaje del usuario menciona un evento importante específico como despido, aumento, ascenso, viaje...?
   -> OBLIGATORIO completar el paso sin excepción:
   -> PASO 1: SIEMPRE usa **save_important_event**.
   -> PASO 2: SIEMPRE usa **update_user_profile** si el evento tiene relación con el perfil del usuario, añadiendo los nuevos datos a los campos y NUNCA eliminando los que ya estaban:
      - relationships: personas y su relación con el usuario ("padres, pareja, amigos relacionados con el evento")
      - hobbies: cosas que le gusta hacer en su tiempo libre, intereses o deportes ("jugar al tenis", "leer", "pintar")
      - goals: objetivos, metas a futuro o deseos mencionados ("quiero cambiar de trabajo", "mejorar mi autoestima", "irme de viaje", "cambiar de vida")
"""

REGLA_SIN_HERRAMIENTAS = """
No tienes ninguna herramienta de memoria disponible en esta conversación.
Responde ÚNICAMENTE con Final Answer, usando solo la información de la sección
"INFORMACIÓN QUE YA TIENES". No intentes llamar a ninguna herramienta bajo ningún concepto.
"""

def build_prompt(memory_enabled: bool, summarizer_enabled: bool, emotion_detection_enabled: bool) -> PromptTemplate:
    ejemplos = []
    reglas = []

    intro = INTRO_CON_EMOCIONES if emotion_detection_enabled else INTRO_SIN_EMOCIONES

    if emotion_detection_enabled:
        reglas.append(REGLA_EMOCIONES)

    if memory_enabled:
        ejemplos.append(EJEMPLO_MEMORIA)
        if summarizer_enabled:
            ejemplos.append(EJEMPLO_HITO_PASO1)
            ejemplos.append(EJEMPLO_HITO_PASO2)
            ejemplos.append(EJEMPLO_HITO_PASO3)
            ejemplos.append(EJEMPLO_HITO_FINAL)
            reglas.append(REGLA_EVENTOS)
            reglas.append(REGLA_MEMORIA)
            reglas.append(REGLA_TOPICS)
        else:
            ejemplos.append(EJEMPLO_HITO_SIN_RESUMEN_PASO1)
            ejemplos.append(EJEMPLO_HITO_SIN_RESUMEN_PASO2)
            ejemplos.append(EJEMPLO_HITO_SIN_RESUMEN_FINAL)
            reglas.append(REGLA_EVENTOS_SIN_RESUMEN)
            reglas.append(REGLA_MEMORIA)
            reglas.append(REGLA_TOPICS_SIN_RESUMEN)
    else:
        ejemplos.append(EJEMPLO_SIN_HERRAMIENTAS)
        reglas.append(REGLA_SIN_HERRAMIENTAS)

    bloque_ejemplos = "\n".join(ejemplos)
    bloque_guia = "GUÍA DE RAZONAMIENTO (Cuándo usar cada herramienta):\n" + "\n".join(reglas)

    texto_completo = intro + PROMPT_HEAD_BODY + bloque_ejemplos + "\n\n" + bloque_guia + PROMPT_TAIL
    return PromptTemplate.from_template(texto_completo)


server_location = {
    "chat_tfg": {
        "command": "python",
        "args": ["server.py"],
        "transport": "stdio"
    }
}

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

if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

async def run_agent(user_input, emotions, memory_enabled=True, summarizer_enabled=True, emotion_detection_enabled=True):
        client = MultiServerMCPClient(server_location)
        mcp_tools = await client.get_tools()
        

        MEMORY_TOOLS = {"save_important_event", "get_important_events",
                "update_user_profile", "get_user_profile"}

        simple_tools = []
        for mcp_tool in mcp_tools:
            def make_wrapper(t):
                async def wrapper(text: str) -> str:
                    try:
                        arg_data = {}
                        if isinstance(text, str) and "{" in text:
                            try:
                                clean_text = text.replace(": None", ": null").replace(":None", ":null")
                                arg_data = json.loads(clean_text)
                            except:
                                arg_data = {"event": text, "query": text}
                        else:
                            arg_data = {"event": text, "query": text}

                        if t.name == "save_important_event":
                            return await t.ainvoke({
                                "session_id": str(st.session_state.session_id),
                                "event": arg_data.get("event", text),
                                "new_type": arg_data.get("new_type", "general"),
                                "importance": arg_data.get("importance", "moderada")
                            })
                        elif t.name == "update_user_profile":
                                def to_str(v):
                                    if isinstance(v, list):
                                        return ", ".join(str(i) for i in v)
                                    return v
                                return await t.ainvoke({
                                    "user_id": str(st.session_state.user_id),
                                    "name": to_str(arg_data.get("name")),
                                    "relationships": to_str(arg_data.get("relationships")),
                                    "goals": to_str(arg_data.get("goals")),
                                    "topics": to_str(arg_data.get("topics")),
                                    "hobbies": to_str(arg_data.get("hobbies"))
                                })
                        elif t.name == "conversation_briefer":
                            return await t.ainvoke({
                                "session_id": str(st.session_state.session_id),
                                "summary": arg_data.get("summary", text)
                            })
                        elif t.name == "get_important_events":
                            return await t.ainvoke({"session_id": str(st.session_state.session_id)})
                        elif t.name == "get_user_profile":
                            return await t.ainvoke({"user_id": str(st.session_state.user_id)})
                        else:
                            return await t.ainvoke({"query": text})

                    except Exception as e:
                        return f"Error en wrapper: {str(e)}"

                return wrapper

            simple_tools.append(Tool(
                name=mcp_tool.name,
                description=mcp_tool.description,
                coroutine=make_wrapper(mcp_tool),
                func=lambda x: x
            ))


        if not memory_enabled:
            simple_tools = [t for t in simple_tools if t.name not in MEMORY_TOOLS]

        SUMMARY_TOOLS = {"conversation_briefer"}
        if not summarizer_enabled:
            simple_tools = [t for t in simple_tools if t.name not in SUMMARY_TOOLS]

        agent_prompt = build_prompt(memory_enabled, summarizer_enabled, emotion_detection_enabled)
        agent = create_react_agent(llm, simple_tools, agent_prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=simple_tools,
            handle_parsing_errors=(
            "Error de formato. Recuerda: NUNCA uses 'Action: Final Answer'. "
            "Para terminar escribe directamente:\n"
            "Thought: Ya tengo suficiente información.\n"
            "Final Answer: [tu respuesta al usuario]"
            ),
            verbose=True,
            max_iterations=10,
        )

        mensajes = db.get_messages(st.session_state.session_id)
        if memory_enabled:
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in mensajes[-20:]])
        else:
            chat_history = ""
        conversation_summ = db.get_conversation_summary(st.session_state.session_id) or ""

        response = await executor.ainvoke({
            "input": user_input,
            "chat_history": chat_history,
            "user_profile": db.get_user_profile_text(st.session_state.user_id) if memory_enabled else "Memoria desactivada.",
            "conversation_summary": conversation_summ if memory_enabled else "",
            "session_id": str(st.session_state.session_id),
            "user_id": str(st.session_state.user_id),
            "emotions": emotions
        })
        return response["output"]

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

    st.divider()
    if st.button("✏️ Nueva conversación", type="primary", use_container_width=True):
        st.session_state.session_id = db.create_session("Nueva conversación", user_id=st.session_state.user_id)
        st.rerun()
    st.divider()
    st.session_state.emotion_detection_enabled = st.toggle(
        "🧠 Detector de emociones",
        value=st.session_state.emotion_detection_enabled,
        help="Activa o desactiva el análisis emocional de tus mensajes."
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

    st.divider()
    st.markdown("**👤 Lo que sé de ti**")
    profile_text = db.get_user_profile_text(st.session_state.user_id)
    

    profile_text = profile_text.replace(".", ". ").replace("  ", " ")
    
    VALORES_VACIOS = {"No indicado", "No indicadas", "No indicados", "Ninguno", "Ninguna", ""}

    if "No hay perfil" in profile_text:
        st.caption("_Todavía no tengo información tuya._")
    else:
        hay_datos = False
        for part in profile_text.split(". "):
            part = part.strip()
            if ": " in part:
                label, value = part.split(": ", 1)
                label = label.strip()
                value = value.strip()
                if value not in VALORES_VACIOS:
                    st.caption(f"**{label}:** {value}")
                    hay_datos = True
        if not hay_datos:
            st.caption("_Todavía no tengo información tuya._")
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

EMOCIONES_POSITIVAS = {
    "admiration", "amusement", "approval", "caring", "curiosity",
    "desire", "excitement", "gratitude", "joy", "love",
    "optimism", "pride", "realization", "relief", "surprise"
}

def tiene_emociones_positivas(emotion_data: list, umbral: float = 0.70) -> bool:
    if not emotion_data:
        return False
    for emo in emotion_data:
        if emo["label"] in EMOCIONES_POSITIVAS and emo["score"] >= umbral:
            return True
    return False


def evaluar_riesgo_crisis(user_input: str, emotion_data: list) -> bool:
    if not user_input.strip():
        return False


    try:
        traduccion = translator(user_input)[0]['translation_text']
        inputs = depression_tokenizer(traduccion, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = depression_model(**inputs).logits
            predicted_class = torch.argmax(logits).item()
        
        if predicted_class == 1:
            if tiene_emociones_positivas(emotion_data, umbral=0.70):
                return False
            else:
                return True


        return predicted_class == 1

    except Exception as e:
        print(f"Error en Capa 2: {e}")
        return True  # fail-safe: ante la duda, activar crisis


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

if st.session_state.crisis_detected:
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


input_value = st.chat_input("", disabled=(st.session_state.crisis_detected or st.session_state.is_thinking))

if input_value:
    st.session_state.pending_input = input_value
    st.session_state.is_thinking = True
    st.rerun()


if st.session_state.pending_input:
    user_input = st.session_state.pending_input

    with st.chat_message("user"):
        st.markdown(user_input)


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
        
    emotion_data = json.loads(emotions)
    scores = {d['label']: d['score'] for d in emotion_data}
    relevant_emotions = emotion_data


    if st.session_state.emotion_detection_enabled and relevant_emotions:
        top = sorted(emotion_data, key=lambda x: x['score'], reverse=True)
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
    

    if evaluar_riesgo_crisis(user_input, emotion_data) and st.session_state.emotion_detection_enabled:
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
                            st.session_state.emotion_detection_enabled
                        ),
                        timeout=60.0
                    ))
                    
                    if not response or not response.strip() or "Agent stopped" in response:
                        raise RuntimeError("Fallo de formato en LangChain")
                        
                except Exception as e:
                    print(f"Error técnico: {e}")
                    response = "Lo siento, me he liado un poco procesando eso. ¿Podrías explicármelo de otra forma?"

                st.markdown(response)
                
        db.save_message(st.session_state.session_id, "assistant", response)
        

        st.session_state.pending_input = None
        st.session_state.is_thinking = False
        st.rerun()