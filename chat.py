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

EMOTION_LABELS = ["anger", "contempt", "disgust", "fear", "frustration","gratitude", "joy", "love", "neutral", "sadness", "surprise"]

llm = ChatOllama(
    model="gemma4:e4b", 
    num_gpu=99,       
    num_ctx=8192)

@st.cache_resource


def load_emotion_detector():
    tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-emotion-classification")
    model = AutoModelForSequenceClassification.from_pretrained("tabularisai/multilingual-emotion-classification")
    model.eval()
    return tokenizer, model

emotion_tokenizer, emotion_model = load_emotion_detector()


def emotion_detector(query: str) -> str:
    inputs = emotion_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=192)
    with torch.no_grad():

        probs = torch.sigmoid(emotion_model(**inputs).logits).cpu().numpy()[0]

    all_emotions = [{"label": EMOTION_LABELS[i], "score": float(probs[i])} for i in range(len(EMOTION_LABELS))]
    all_emotions.sort(key=lambda x: x["score"], reverse=True)
    
    result = [e for e in all_emotions if e["score"] >= 0.1]
    
        
    return json.dumps(result)


INTRO_CON_EMOCIONES = """
Eres un asistente de chat que genera conversaciones casuales y realistas. No des respuestas largas, que sea una conversación humana. 
En caso de detectar que el usuario necesita de validación empática, comprende sus emociones y responde acorde a la situación que te planteen, pero siempre de forma casual. En caso contrario, limítate a mantener una conversación realista y breve.
Para saber cómo se siente el usuario, dispones de la confianza con la que pueden estar ciertas emociones en el mensaje; {emotions}. A su vez, si existe un {conversation_summary}, úsalo para recordar los eventos 
importantes que ya han ocurrido en esta sesión.
"""

INTRO_SIN_EMOCIONES = """
Eres un asistente de chat que genera conversaciones casuales y realistas. No des respuestas largas, que sea una conversación humana. 
En caso de detectar que el usuario necesita de validación empática, comprende el contexto de lo que dice y responde acorde a la situación que te plantee, pero siempre de forma casual. En caso contrario, limítate a mantener una conversación realista y breve.
Si existe un {conversation_summary}, úsalo para recordar los eventos importantes que ya han ocurrido en esta sesión.
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

EJEMPLO_HITO = """
EJEMPLO DE HITO VITAL (OBLIGATORIO SEGUIR ESTE FORMATO):
Usuario: "Me han ascendido en el trabajo"
Thought: Hito vital detectado. Debo completar los pasos obligatorios.
Action: save_important_event
Action Input: {{"session_id": "{session_id}", "event": "Ascenso laboral", "new_type": "logro", "importance": "alta"}}
Observation: Evento guardado.
Thought: Ahora debo guardar el resumen del evento.
Action: conversation_briefer
Action Input: {{"session_id": "{session_id}", "summary": "El usuario ha recibido un ascenso laboral. Estado emocional positivo y celebratorio."}}
Observation: Resumen actualizado.
Thought: Ahora debo actualizar el perfil.
Action: update_user_profile
Action Input: {{"user_id": "{user_id}", "topics": "logro profesional, ascenso laboral"}}
Observation: Perfil actualizado.
Thought: Los pasos completados. Ahora respondo.
Final Answer: ¡Enhorabuena! Eso es una gran noticia...
"""

EJEMPLO_HITO_SIN_RESUMEN = """
EJEMPLO DE HITO VITAL (OBLIGATORIO SEGUIR ESTE FORMATO):
Usuario: "Me han ascendido en el trabajo"
Thought: Hito vital detectado. Debo completar los pasos obligatorios.
Action: save_important_event
Action Input: {{"session_id": "{session_id}", "event": "Ascenso laboral", "new_type": "logro", "importance": "alta"}}
Observation: Evento guardado.
Thought: Ahora debo actualizar el perfil.
Action: update_user_profile
Action Input: {{"user_id": "{user_id}", "topics": "logro profesional, ascenso laboral"}}
Observation: Perfil actualizado.
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
   ¿El usuario menciona personas de su vida (familia, pareja, amigos, compañeros)?
   ¿Habla de algo que quiere conseguir, cambiar o mejorar?
   ¿Menciona objetivos, metas a futuro o deseos que quiera ver cumplidos?
   -> USA **get_user_profile** para ver si lo que ha mencionado está ya en la base de datos. En caso contrario usa **update_user_profile** para añadir los nuevos datos.
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
4. ¿El usuario pregunta por algo que te contó en esta sesión o quieres recordar eventos previos de la misma? 
   -> Usa **get_important_events**, **get_user_profile** y {conversation_summary} para recordar dicha información.
   -> Usa **update_user_profile** para guardarlo como tema recurrente:
        - topics: temas recurrentes ("estrés laboral", "problemas de pareja", "problemas en el trabajo")
"""

REGLA_TOPICS_SIN_RESUMEN = """
4. ¿El usuario pregunta por algo que te contó en esta sesión o quieres recordar eventos previos de la misma? 
   -> Usa **get_important_events** y **get_user_profile** para recordar dicha información.
   -> Usa **update_user_profile** para guardarlo como tema recurrente:
        - topics: temas recurrentes ("estrés laboral", "problemas de pareja", "problemas en el trabajo")
"""

REGLA_EVENTOS = """
3. ¿El usuario menciona un evento importante específico como despido, aumento, ascenso, viaje...?
   -> OBLIGATORIO completar los pasos sin excepción, en este orden:
   -> PASO 1: SIEMPRE usa **save_important_event** y **conversation_briefer** en ese orden.
   -> PASO 2: SIEMPRE usa **update_user_profile** con los campos relevantes del evento:
        - relationships: personas y su relación con el usuario ("madre sobreprotectora", "mejor amigo Carlos")
        - hobbies: cosas que le gusta hacer en su tiempo libre, intereses o deportes ("jugar al tenis", "leer", "pintar")
        - goals: objetivos, metas a futuro o deseos mencionados ("quiero cambiar de trabajo", "mejorar mi autoestima", "irme de viaje", "cambiar de vida")
"""

REGLA_EVENTOS_SIN_RESUMEN = """
3. ¿El usuario menciona un evento importante específico como despido, aumento, ascenso, viaje...?
   -> OBLIGATORIO completar el paso sin excepción:
   -> PASO 1: SIEMPRE usa **save_important_event**.
   -> PASO 2: SIEMPRE usa **update_user_profile** con los campos relevantes del evento:
      - relationships: personas y su relación con el usuario ("madre sobreprotectora", "mejor amigo Carlos")
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
            ejemplos.append(EJEMPLO_HITO)
            reglas.append(REGLA_EVENTOS)
            reglas.append(REGLA_MEMORIA)
            reglas.append(REGLA_TOPICS)
        else:
            ejemplos.append(EJEMPLO_HITO_SIN_RESUMEN)
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
            max_iterations=10
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


PATRONES_IMPLICITOS = [

    r"no quiero (seguir|estar aquí|vivir|continuar|existir)",
    r"quiero (desaparecer|dormir para siempre|no despertar|morir|quitarme la vida|acabar con mi vida)",
    r"(pienso|estoy pensando) en (suicidarme|quitarme la vida|hacerme daño|no seguir)",
    r"voy a (acabar|terminar) con (todo|mi vida|esto de una vez)",
 

    r"(todo|nada) (sería|seria|estaría|estaria) mejor si (yo )?(no estuviera|desapareciera|no existiera)",
    r"no hay (salida|solución|esperanza|remedio|nada que hacer)",
    r"nunca (va a|voy a) (mejorar|cambiar|estar bien|ponerse bien)",
    r"todo (está|ha) (perdido|acabado)",
    r"(ya )?no (me) queda (nada|nadie|ninguna razón) (para seguir|para vivir|para continuar)",

    r"ya no (puedo|aguanto|soporto) (más|esto|nada)",
    r"estoy (harto|harta|cansado|cansada) de (vivir|todo|seguir|luchar|intentarlo)",
    r"no (puedo|quiero) (seguir|continuar) (así|más|adelante)",
    r"no tengo (fuerzas|energía|ganas) (para|de) (seguir|continuar|nada|vivir)",

    r"(qué|que) sentido tiene (vivir|todo|esto|seguir)",
    r"para qué (sirvo|sigo|vivo|seguir|vivir|continuar|intentarlo)",
    r"no (tiene|hay) ningún (sentido|motivo|razón) para (seguir|vivir|continuar)",

    r"a nadie le importa(ría)? (si|que)",
    r"soy una carga (para|a)",
    r"(nadie|todos) (estarían|estarian|estaría|estaria|serían|serian) mejor sin (mí|mi)",
    r"el mundo (estaría|estaria|sería|seria) mejor sin (mí|mi)",
    r"no merezco (vivir|estar aquí|seguir|nada)",

    r"no voy a (estar|seguir) (aquí|mañana)",
    r"(cuídate|cuídense|cuídate mucho) (cuando )?(yo )?(no esté|me haya ido|ya no esté)",
    r"(esta es mi )?(última|ultima) (vez|oportunidad|noche|semana|conversación)",
    r"quiero (despedirme|decirte adiós|deciros adiós)",

    r"(me rindo|doy por vencido|doy por vencida)",
    r"ya no (voy a|quiero) (intentarlo|luchar|pelear|seguir intentando)",

    r"quiero (hacerme daño|lastimarme|herirme|dañarme)",
    r"me (he|voy a) (hecho daño|lastimado|herido)",
 

    r"(tengo|he preparado|tengo pensado) (un plan|cómo hacerlo|todo preparado|todo listo)",
    r"(ya )?(sé|se) (cómo|como) (hacerlo|terminar|acabar con todo)",
 

    r"no (me) veo (en el futuro|mañana|la semana que viene|el año que viene)",
    r"no (habrá|hay) (un )?(mañana|futuro) (para mí|para mi)",
]

EMOCIONES_NEGATIVAS = {"anger", "contempt", "disgust", "fear", "frustration", "sadness"}

def emocion_primaria_negativa(emotion_data: list) -> bool:
    if not emotion_data:
        return False
    top = max(emotion_data, key=lambda x: x["score"])
    return top["label"] in EMOCIONES_NEGATIVAS

def evaluar_riesgo_crisis(user_input: str, emotion_data: list) -> bool:
    if not user_input.strip():
        return False


    if not emocion_primaria_negativa(emotion_data):
        return False

    texto = user_input.lower()
    for patron in PATRONES_IMPLICITOS:
        if re.search(patron, texto):
            return True

    return False


EMOJI_MAP = {
"anger": "😠",
"contempt": "😒",
"disgust": "🤢",
"fear": "😨",
"frustration": "😤",
"gratitude": "🙏",
"joy": "😊",
"love": "❤️",
"neutral": "😐",
"sadness": "😢",
"surprise": "😲"
}


EN_TO_ES_MAP = {
"anger": "Rabia",
"contempt": "Desprecio",
"disgust": "Asco",
"fear": "Miedo",
"frustration": "Frustración",
"gratitude": "Gratitud",
"joy": "Alegría",
"love": "Amor",
"neutral": "Neutral",
"sadness": "Tristeza",
"surprise": "Sorpresa"
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

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.emotion_detection_enabled:
        emotions = emotion_detector(user_input)
    else:
        emotions = json.dumps([])
    db.save_message(st.session_state.session_id, "user", user_input, emotions)
    mensajes = db.get_messages(st.session_state.session_id)
    if len(mensajes) == 1:
        db.update_session(st.session_state.session_id, new_name=user_input[:50])
    emotion_data = json.loads(emotions)
    scores = {d['label']: d['score'] for d in emotion_data}
    relevant_emotions = emotion_data


    if st.session_state.emotion_detection_enabled and relevant_emotions:
        top = sorted(emotion_data, key=lambda x: x['score'], reverse=True)
        with st.markdown("🔍 Emociones detectadas"):
            st.caption("ℹ️ *Los porcentajes indican la confianza de que esa emoción esté presente en el mensaje. Al ser independientes, la suma puede superar el 100%.*")
            
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
    
    if evaluar_riesgo_crisis(user_input, emotion_data):
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
                "Estamos juntos en esto.""")
        db.save_message(st.session_state.session_id, "assistant", "ALERTA DE CRISIS: El modelo ha detectado un riesgo crítico de depresión/crisis en el mensaje del usuario.")
        st.rerun()
    else:

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
                st.rerun()
                
        except Exception as e:

            print(f"Error técnico: {e}")
            response = "Lo siento, me he liado un poco procesando eso. ¿Podrías explicármelo de otra forma?"

        with st.chat_message("assistant"):
            st.markdown(response)
        db.save_message(st.session_state.session_id, "assistant", response)
        st.rerun()