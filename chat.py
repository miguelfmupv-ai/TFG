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
    resumenes = [
        db.get_conversation_summary(s["id"])
        for s in sessions
        if s["user_id"] == user_id
    ]
    resumenes = [r.strip() for r in resumenes if r and r.strip()]

    if not resumenes:
        return ""

    ultimos = list(reversed(resumenes))

    return " | ".join(ultimos)



# ──────────────────────────────────────────────────────────────────────────
# BLOQUE FIJO — no debe contener variables que cambien turno a turno
# ──────────────────────────────────────────────────────────────────────────

INTRO_CON_EMOCIONES = """
Eres un asistente de chat que genera conversaciones casuales y realistas, como hablarías con un conocido cercano por mensaje. No uses un tono excesivamente cariñoso, efusivo ni "terapéutico" por defecto — adapta tu tono al del usuario. Si el usuario escribe de forma seca o breve, responde igual de conciso. Si escribe de forma animada, puedes ser más expresivo, pero sin forzarlo.
No des respuestas largas. Evita frases como "estoy aquí para ti" o "no importa si hablas o no" salvo que el contexto emocional sea claramente grave.
En caso de detectar que el usuario necesita validación empática real (no cualquier emoción negativa leve), compréndela y responde acorde, pero de forma natural y breve, sin sonar como un terapeuta.
Dispones de información contextual sobre el usuario (perfil, emociones detectadas, resumen histórico, predicción emocional) en la sección "INFORMACIÓN DE ESTE TURNO" al final de este prompt. Si detectas un evento importante, habla de alguna persona en su vida, o menciona algún objetivo o meta a conseguir, debes guardarlo usando las herramientas disponibles y basándote en las reglas de razonamiento.
"""

INTRO_SIN_EMOCIONES = """
Eres un asistente de chat que genera conversaciones casuales y realistas, como hablarías con un conocido cercano por mensaje. No uses un tono excesivamente cariñoso, efusivo ni "terapéutico" por defecto — adapta tu tono al del usuario.
No des respuestas largas. En caso de detectar que el usuario necesita de validación empática real, comprende el contexto y responde acorde, pero siempre de forma casual y breve.
Dispones de información contextual sobre el usuario (perfil, resumen histórico) en la sección "INFORMACIÓN DE ESTE TURNO" al final de este prompt. Si detectas un evento importante, habla de alguna persona en su vida, o menciona algún objetivo o meta a conseguir, debes guardarlo usando las herramientas disponibles y basándote en las reglas de razonamiento.
"""

CON_PREDICCIÓN = """
También dispones de una predicción de la distribución emocional del usuario basada en sus sesiones anteriores, indicada en "INFORMACIÓN DE ESTE TURNO".
   -> Úsala como contexto previo antes de leer su mensaje, no como un hecho certero.
   -> Si la predicción apunta a emociones negativas y el mensaje lo confirma, adapta el tono de respuesta según las reglas anteriores.
   -> No menciones la predicción al usuario bajo ningún concepto.
"""

PROMPT_HEAD_FIJO = """
INSTRUCCIÓN GENERAL: Si el perfil biográfico incluido en la información de este turno contiene un nombre real (distinto de "No indicado"), y el usuario te saluda de forma genérica ("hola", "qué tal"), usa su nombre en tu respuesta de forma natural.

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

Cuando NO necesitas herramientas (ya tienes toda la info en la sección INFORMACIÓN DE ESTE TURNO al final):

Thought: [tu razonamiento]
Final Answer: [tu respuesta al usuario]

CRÍTICO — Sobre el uso de Action:
- Action: SÍ se usa para llamar herramientas de {tool_names}
- Action: NUNCA se usa para "Final Answer" — "Final Answer" NO es una herramienta
- PROHIBIDO:  Action: Final Answer
- CORRECTO:   Final Answer: [tu respuesta]
- Después del último Thought escribe siempre "Final Answer:" directamente, sin Action.

IMPORTANTE — Uso de identificadores:
- El ID de sesión y el ID de usuario necesarios para las herramientas se encuentran en la sección "INFORMACIÓN DE ESTE TURNO", al final de este prompt. Úsalos siempre literalmente tal como aparecen ahí, nunca los inventes.
- Para guardar datos de perfil (nombre, relaciones, objetivos, temas, aficiones), usa siempre el ID de usuario, NUNCA el ID de sesión.
- Para guardar eventos o resúmenes de sesión, usa siempre el ID de sesión.
"""

# ──────────────────────────────────────────────────────────────────────────
# EJEMPLOS — fijos, sin variables interpoladas
# ──────────────────────────────────────────────────────────────────────────

EJEMPLO_SIN_HERRAMIENTAS = """
EJEMPLO DE RESPUESTA SIN HERRAMIENTAS:
Usuario: "¿Cómo estás?"
Thought: Es una pregunta casual, tengo toda la info necesaria en la sección de información de este turno. No necesito herramientas.
Final Answer: Todo bien por aquí, ¿qué tal tú?
"""

EJEMPLO_MEMORIA = """
EJEMPLO DE USO CORRECTO DE HERRAMIENTAS:
Usuario: "Me llamo Ana y quiero mejorar mi autoestima"
Thought: El usuario menciona un objetivo personal. Debo usar update_user_profile con el ID de usuario indicado en la información de este turno.
Action: update_user_profile
Action Input: {{"user_id": "<ID_USUARIO_DE_ESTE_TURNO>", "name": "Ana", "goals": "mejorar autoestima"}}
Observation: Perfil actualizado.
Thought: Ya tengo suficiente información para responder.
Final Answer: Encantada de conocerte, Ana. Trabajar en la autoestima es un objetivo muy valioso...
"""

EJEMPLO_HITO_PASO1 = """
EJEMPLO (PASO 1 de un hito vital — guardar el evento):
Usuario: "Me han ascendido en el trabajo"
Thought: Hito vital detectado. Primer paso obligatorio: guardar el evento usando el ID de sesión indicado en la información de este turno.
Action: save_important_event
Action Input: {{"session_id": "<ID_SESION_DE_ESTE_TURNO>", "event": "Ascenso laboral", "new_type": "logro", "importance": "alta"}}
"""

EJEMPLO_HITO_PASO2 = """
EJEMPLO (PASO 2 — tras recibir "Evento guardado." como Observation):
Thought: El evento ya está guardado. Ahora debo resumir la conversación.
Action: conversation_briefer
Action Input: {{"session_id": "<ID_SESION_DE_ESTE_TURNO>", "summary": "El usuario ha recibido un ascenso laboral. Estado emocional positivo y celebratorio."}}
"""

EJEMPLO_HITO_PASO3 = """
EJEMPLO (PASO 3 — tras recibir "Resumen guardado." como Observation):
Thought: Evento y resumen guardados. Ahora actualizo el perfil con el nuevo logro, usando el ID de usuario.
Action: update_user_profile
Action Input: {{"user_id": "<ID_USUARIO_DE_ESTE_TURNO>", "topics": "logro profesional, ascenso laboral"}}
"""

EJEMPLO_HITO_FINAL = """
EJEMPLO (PASO FINAL — tras recibir "Perfil actualizado." como Observation):
Thought: Los 3 pasos obligatorios están completos. Ahora respondo al usuario.
Final Answer: ¡Enhorabuena! Eso es una gran noticia...
"""

EJEMPLO_HITO_SIN_RESUMEN_PASO1 = """
EJEMPLO DE HITO VITAL (OBLIGATORIO SEGUIR ESTE FORMATO):
Usuario: "Me han ascendido en el trabajo"
Thought: Hito vital detectado. Debo completar los pasos obligatorios usando el ID de sesión indicado en la información de este turno.
Action: save_important_event
Action Input: {{"session_id": "<ID_SESION_DE_ESTE_TURNO>", "event": "Ascenso laboral", "new_type": "logro", "importance": "alta"}}
"""

EJEMPLO_HITO_SIN_RESUMEN_PASO2 = """
EJEMPLO (PASO 2 — tras recibir "Evento guardado." como Observation):
Thought: Ahora debo actualizar el perfil usando el ID de usuario.
Action: update_user_profile
Action Input: {{"user_id": "<ID_USUARIO_DE_ESTE_TURNO>", "topics": "logro profesional, ascenso laboral"}}
"""

EJEMPLO_HITO_SIN_RESUMEN_FINAL = """
EJEMPLO (PASO FINAL — tras recibir "Perfil actualizado." como Observation):
Thought: Los pasos completados. Ahora respondo.
Final Answer: ¡Enhorabuena! Eso es una gran noticia...
"""

# ──────────────────────────────────────────────────────────────────────────
# REGLAS — fijas, sin variables interpoladas, referencian la sección final
# ──────────────────────────────────────────────────────────────────────────

REGLA_EMOCIONES = """
0. Antes de responder, revisa las emociones detectadas indicadas en la información de este turno:
   -> Si la emoción dominante es negativa (sadness, anger, fear, frustration, disgust, contempt)
      con confianza alta (>60%), tu Final Answer debe reconocer ese estado emocional de forma
      breve y natural antes de continuar la conversación, SIN convertir la respuesta en una pregunta terapéutica tipo "¿cómo te sientes?" salvo que el usuario lo pida explícitamente.
   -> Si la emoción dominante es positiva, neutral, o negativa pero con confianza baja/media (<60%),
      no fuerces empatía ni validación emocional: responde de forma casual y directa, acompañando el tono del usuario.
   -> No menciones los porcentajes ni el nombre técnico de la emoción al usuario.
   -> Evita frases de acompañamiento genéricas ("estoy aquí para ti", "no importa si hablas o no") salvo que el contexto sea claramente grave.
"""

REGLA_MEMORIA = """
1. ¿El usuario te dice su nombre?
   ¿El mensaje del usuario menciona personas de su vida con las que haya interactuado o especifica relaciones sociales (familia, pareja, amigos, compañeros)?
   ¿Habla de algo que quiere conseguir, cambiar o mejorar?
   ¿Menciona objetivos, metas a futuro o deseos que quiera ver cumplidos?
   -> USA **get_user_profile** para ver si lo que ha mencionado está ya en la base de datos. En caso contrario usa **update_user_profile** para añadir los nuevos datos y NUNCA eliminando los que ya estaban.
    Usa siempre el ID de usuario indicado en la información de este turno como "user_id", NUNCA el ID de sesión:
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
4. ¿El mensaje del usuario pregunta por algo que te contó en esta sesión o quieres recordar eventos previos de la misma? 
   -> Usa **get_important_events**, **get_user_profile** y el resumen histórico indicado en la información de este turno para recordar dicha información.
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
   -> PASO 1: SIEMPRE usa **save_important_event** y **conversation_briefer** en ese orden, usando el ID de sesión indicado en la información de este turno.
   -> PASO 2: SIEMPRE usa **update_user_profile** (con el ID de usuario) si el evento tiene relación con el perfil del usuario, añadiendo los nuevos datos a los campos y NUNCA eliminando los que ya estaban:
        - relationships: personas y su relación con el usuario ("padres, pareja, amigos relacionados con el evento")
        - hobbies: cosas que le gusta hacer en su tiempo libre, intereses o deportes ("jugar al tenis", "leer", "pintar")
        - goals: objetivos, metas a futuro o deseos mencionados ("quiero cambiar de trabajo", "mejorar mi autoestima", "irme de viaje", "cambiar de vida")
"""

REGLA_EVENTOS_SIN_RESUMEN = """
3. ¿El mensaje del usuario menciona un evento importante específico como despido, aumento, ascenso, viaje...?
   -> OBLIGATORIO completar el paso sin excepción:
   -> PASO 1: SIEMPRE usa **save_important_event** con el ID de sesión indicado en la información de este turno.
   -> PASO 2: SIEMPRE usa **update_user_profile** (con el ID de usuario) si el evento tiene relación con el perfil del usuario, añadiendo los nuevos datos a los campos y NUNCA eliminando los que ya estaban:
      - relationships: personas y su relación con el usuario ("padres, pareja, amigos relacionados con el evento")
      - hobbies: cosas que le gusta hacer en su tiempo libre, intereses o deportes ("jugar al tenis", "leer", "pintar")
      - goals: objetivos, metas a futuro o deseos mencionados ("quiero cambiar de trabajo", "mejorar mi autoestima", "irme de viaje", "cambiar de vida")
"""

REGLA_SIN_HERRAMIENTAS = """
No tienes ninguna herramienta de memoria disponible en esta conversación, y no dispones de ningún dato biográfico del usuario (nombre, relaciones, etc).
Responde ÚNICAMENTE con Final Answer. NUNCA menciones que no tienes su nombre o datos — simplemente no los uses ni los menciones. No intentes llamar a ninguna herramienta bajo ningún concepto.
"""

# ──────────────────────────────────────────────────────────────────────────
# BLOQUE VARIABLE — todo lo que cambia turno a turno va al final
# ──────────────────────────────────────────────────────────────────────────

PROMPT_TAIL_VARIABLE = """

INFORMACIÓN DE ESTE TURNO:
- Perfil biográfico actual: {user_profile}
- Emociones principales actuales: {emotions}
- Resumen histórico: {conversation_summary}
- ID de sesión actual: {session_id}
- ID de usuario actual: {user_id}
- Emoción predominante predicha para esta sesión: {predicted_emotion}

Historial de conversación:
{chat_history}

Mensaje del usuario: {input}

{agent_scratchpad}"""


# ──────────────────────────────────────────────────────────────────────────
# build_prompt actualizado
# ──────────────────────────────────────────────────────────────────────────

PROMPT_GESTION_PERFIL = """
Eres un asistente cuya única función es gestionar el perfil del usuario (añadir, modificar, eliminar un valor concreto, o borrar un campo entero) cuando te lo pida.

HERRAMIENTAS DISPONIBLES:
{tools}

Nombres de herramientas disponibles: {tool_names}

IMPORTANTE: El "ID de usuario actual" aparece más abajo en este prompt. Cópialo EXACTAMENTE tal cual, nunca escribas null ni lo inventes.

REGLAS DE FORMATO (OBLIGATORIO):
- Si pide borrar TODO un campo de golpe, usa action='clear' (sin 'value').
- Si pide eliminar solo un valor concreto, usa action='remove' con 'value'.
- Si pide añadir algo, usa action='add' con 'value'.
- Si pide sustituir un valor por otro, usa action='modify' con 'old_value' y 'new_value'.
- Tras recibir el Observation de la herramienta, SIEMPRE debes terminar con Thought + Final Answer. NUNCA repitas la misma Action de nuevo.

EJEMPLO (añadir):
Usuario: "añade que me gusta la fotografía"
Thought: El usuario quiere añadir una nueva afición. Debo usar action='add'.
Action: edit_profile_value
Action Input: {{"user_id": "{user_id}", "field": "aficiones", "action": "add", "value": "fotografía"}}
Observation: Perfil actualizado.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He añadido la fotografía a tus aficiones.

EJEMPLO (eliminar un valor concreto):
Usuario: "borra que me gusta dibujar"
Thought: El usuario quiere eliminar un valor concreto del campo aficiones. Debo usar action='remove'.
Action: edit_profile_value
Action Input: {{"user_id": "{user_id}", "field": "aficiones", "action": "remove", "value": "dibujar"}}
Observation: Perfil actualizado.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He eliminado "dibujar" de tus aficiones.

EJEMPLO (modificar):
Usuario: "cambia mi objetivo de viajar al Caribe por viajar a Japón"
Thought: El usuario quiere sustituir un objetivo por otro. Debo usar action='modify'.
Action: edit_profile_value
Action Input: {{"user_id": "{user_id}", "field": "objetivos", "action": "modify", "old_value": "viajar al Caribe", "new_value": "viajar a Japón"}}
Observation: Perfil actualizado.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He actualizado tu objetivo a viajar a Japón.

EJEMPLO (borrado total de un campo):
Usuario: "borra todos mis temas"
Thought: El usuario pide borrar el campo completo de temas. Debo usar action='clear'.
Action: edit_profile_value
Action Input: {{"user_id": "{user_id}", "field": "temas", "action": "clear"}}
Observation: Campo eliminado por completo.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He borrado todos tus temas recurrentes.

Campos válidos: nombre, relaciones, objetivos, temas, aficiones.

CRÍTICO:
- NUNCA uses Action: Final Answer
- Escribe siempre Final Answer: directamente tras el último Thought, SIN repetir la misma Action de nuevo.

Mensaje del usuario: {input}

{agent_scratchpad}
"""

PROMPT_GESTION_EVENTOS = """
Eres un asistente cuya única función es gestionar los eventos de la sesión (añadir, modificar, eliminar uno concreto, o borrarlos todos) cuando el usuario te lo pida.

HERRAMIENTAS DISPONIBLES:
{tools}

Nombres de herramientas disponibles: {tool_names}

IMPORTANTE: El "ID de sesión actual" aparece más abajo en este prompt. Cópialo EXACTAMENTE tal cual, nunca escribas null ni lo inventes.

IMPORTANTE: Si el usuario menciona algo que quiere guardar como evento (incluyendo lesiones, accidentes, o situaciones ya ocurridas y sin urgencia presente), trátalo como una petición normal de guardar un evento usando la herramienta correspondiente. Solo evita usar las herramientas si la persona expresa una emergencia activa que requiere ayuda inmediata, en cuyo caso prioriza remitirle a ayuda profesional antes que ejecutar cualquier acción.

REGLAS DE FORMATO (OBLIGATORIO):
- Si pide borrar TODOS los eventos de golpe, usa action='clear' (sin más parámetros).
- Para eliminar o modificar UN evento concreto, primero consulta get_important_events para obtener el event_id.
- Usa action='add' para añadir, 'remove' para eliminar uno, 'modify' para modificar uno.
- Tras recibir el Observation de la herramienta, SIEMPRE debes terminar con Thought + Final Answer. NUNCA repitas la misma Action de nuevo.

EJEMPLO (añadir evento):
Usuario: "añade que me caí en el trabajo"
Thought: El usuario quiere guardar un evento sobre una situación ya ocurrida, sin urgencia activa. Debo usar action='add'.
Action: edit_event
Action Input: {{"session_id": "{session_id}", "action": "add", "event": "Se cayó en el trabajo", "new_type": "salud", "new_importance": "moderada"}}
Observation: Evento añadido.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He guardado el evento de tu caída en el trabajo.

EJEMPLO (eliminar un evento concreto):
Usuario: "borra el evento del ascenso"
Thought: Necesito el event_id del evento del ascenso. Primero consulto los eventos disponibles.
Action: get_important_events
Action Input: {{"session_id": "{session_id}"}}
Observation: [{{"id": "abc123", "event": "Ascenso laboral", "date": "..."}}]
Thought: Ya tengo el event_id. Ahora elimino ese evento concreto.
Action: edit_event
Action Input: {{"session_id": "{session_id}", "action": "remove", "event_id": "abc123"}}
Observation: Evento eliminado.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He eliminado el evento del ascenso.

EJEMPLO (modificar un evento):
Usuario: "cambia la importancia del evento del ascenso a media"
Thought: Necesito el event_id del evento del ascenso. Primero consulto los eventos disponibles.
Action: get_important_events
Action Input: {{"session_id": "{session_id}"}}
Observation: [{{"id": "abc123", "event": "Ascenso laboral", "date": "..."}}]
Thought: Ya tengo el event_id. Ahora modifico su importancia.
Action: edit_event
Action Input: {{"session_id": "{session_id}", "action": "modify", "event_id": "abc123", "new_importance": "media"}}
Observation: Evento modificado.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He actualizado la importancia del evento del ascenso a media.

EJEMPLO (borrado total):
Usuario: "borra todos los eventos de esta sesión"
Thought: El usuario pide eliminar todos los eventos. Debo usar action='clear'.
Action: edit_event
Action Input: {{"session_id": "{session_id}", "action": "clear"}}
Observation: Todos los eventos de la sesión han sido eliminados.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He eliminado todos los eventos de esta sesión.

CRÍTICO:
- NUNCA uses Action: Final Answer
- Escribe siempre Final Answer: directamente tras el último Thought, SIN repetir la misma Action de nuevo.

Mensaje del usuario: {input}

{agent_scratchpad}
"""

PROMPT_GESTION_RESUMEN = """
Eres un asistente cuya única función es gestionar el resumen de la sesión actual (añadir, modificar, eliminar contenido concreto, o borrarlo por completo) cuando el usuario te lo pida.

HERRAMIENTAS DISPONIBLES:
{tools}

Nombres de herramientas disponibles: {tool_names}

IMPORTANTE: El "ID de sesión actual" aparece más abajo en este prompt. Cópialo EXACTAMENTE tal cual, nunca escribas null ni lo inventes.

REGLAS DE FORMATO (OBLIGATORIO):
- Si pide borrar TODO el resumen, usa action='clear' (sin 'value').
- Si pide eliminar solo una parte concreta, usa action='remove' con 'value'.
- Si pide añadir algo, usa action='add' con 'value'.
- Si pide sustituir un contenido por otro, usa action='modify' con 'old_value' y 'new_value'.
- Tras recibir el Observation de la herramienta, SIEMPRE debes terminar con Thought + Final Answer. NUNCA repitas la misma Action de nuevo.

EJEMPLO (añadir):
Usuario: "añade que también hablamos de mi ansiedad por los exámenes"
Thought: El usuario quiere añadir un fragmento al resumen. Debo usar action='add'.
Action: edit_session_summary
Action Input: {{"session_id": "{session_id}", "action": "add", "value": "El usuario mencionó ansiedad por los exámenes"}}
Observation: Resumen actualizado.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He añadido esa parte al resumen de la sesión.

EJEMPLO (eliminar una parte):
Usuario: "elimina la parte sobre el accidente laboral"
Thought: El usuario quiere eliminar un fragmento concreto. Debo usar action='remove'.
Action: edit_session_summary
Action Input: {{"session_id": "{session_id}", "action": "remove", "value": "accidente laboral"}}
Observation: Resumen actualizado.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He eliminado esa parte del resumen.

EJEMPLO (modificar):
Usuario: "cambia la parte del ascenso por que fue un ascenso a gerente"
Thought: El usuario quiere sustituir contenido del resumen. Debo usar action='modify'.
Action: edit_session_summary
Action Input: {{"session_id": "{session_id}", "action": "modify", "old_value": "ascenso", "new_value": "ascenso a gerente"}}
Observation: Resumen actualizado.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He actualizado esa parte del resumen.

EJEMPLO (borrado total):
Usuario: "borra todo el resumen"
Thought: El usuario pide borrar el resumen completo. Debo usar action='clear'.
Action: edit_session_summary
Action Input: {{"session_id": "{session_id}", "action": "clear"}}
Observation: Resumen eliminado por completo.
Thought: Ya he completado la acción solicitada, no necesito volver a llamar a la herramienta.
Final Answer: He borrado el resumen de esta sesión.

CRÍTICO:
- NUNCA uses Action: Final Answer
- Escribe siempre Final Answer: directamente tras el último Thought, SIN repetir la misma Action de nuevo.

Mensaje del usuario: {input}

{agent_scratchpad}
"""


def build_prompt(memory_enabled: bool, summarizer_enabled: bool, emotion_detection_enabled: bool, emotion_prediction_enabled: bool) -> PromptTemplate:
    ejemplos = []
    reglas = []

    intro = INTRO_CON_EMOCIONES if emotion_detection_enabled else INTRO_SIN_EMOCIONES

    if emotion_prediction_enabled:
        intro += CON_PREDICCIÓN

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

    # Orden: fijo (intro + head + ejemplos + reglas) -> variable (al final)
    texto_completo = intro + PROMPT_HEAD_FIJO + bloque_ejemplos + "\n\n" + bloque_guia + PROMPT_TAIL_VARIABLE
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

if "emotion_prediction_enabled" not in st.session_state:
    st.session_state.emotion_prediction_enabled = True

if "reasoning_enabled" not in st.session_state:
    st.session_state.reasoning_enabled = True

def get_llm(reasoning: bool):
    return ChatOllama(
        model="gemma4:e4b",
        num_gpu=99,
        num_ctx=8192,
        reasoning=reasoning
    )

if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

async def run_agent(user_input, emotions, memory_enabled=True, summarizer_enabled=True, emotion_detection_enabled=True, predicted_emotion=None, emotion_prediction_enabled=True):
        client = MultiServerMCPClient(server_location)
        try:
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
                                result = await t.ainvoke({
                                    "session_id": str(st.session_state.session_id),
                                    "event": arg_data.get("event", text),
                                    "new_type": arg_data.get("new_type", "general"),
                                    "importance": arg_data.get("importance", "moderada")
                                })
                                if isinstance(result, list):
                                    return result[0].get("text", str(result))
                                return str(result)
                            elif t.name == "update_user_profile":
                                    def to_str(v):
                                        if isinstance(v, list):
                                            return ", ".join(str(i) for i in v)
                                        return v
                                    result = await t.ainvoke({
                                        "user_id": str(st.session_state.user_id),
                                        "name": to_str(arg_data.get("name")),
                                        "relationships": to_str(arg_data.get("relationships")),
                                        "goals": to_str(arg_data.get("goals")),
                                        "topics": to_str(arg_data.get("topics")),
                                        "hobbies": to_str(arg_data.get("hobbies"))
                                    })
                                    if isinstance(result, list):
                                        return result[0].get("text", str(result))
                                    return str(result)
                            elif t.name == "conversation_briefer":
                                result = await t.ainvoke({
                                    "session_id": str(st.session_state.session_id),
                                    "summary": arg_data.get("summary", text)
                                })
                                if isinstance(result, list):
                                    return result[0].get("text", str(result))
                                return str(result)
                            elif t.name == "get_important_events":
                                result = await t.ainvoke({"session_id": str(st.session_state.session_id)})
                                if isinstance(result, list):
                                    return result[0].get("text", str(result))
                                return str(result)
                            elif t.name == "get_user_profile":
                                result = await t.ainvoke({"user_id": str(st.session_state.user_id)})
                                if isinstance(result, list):
                                    return result[0].get("text", str(result))
                                return str(result)
                            else:
                                result = await t.ainvoke({"query": text})
                                if isinstance(result, list):
                                    return result[0].get("text", str(result))
                                return str(result)

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

            agent_prompt = build_prompt(memory_enabled, summarizer_enabled, emotion_detection_enabled, emotion_prediction_enabled)
            llm = get_llm(st.session_state.reasoning_enabled)
            agent = create_react_agent(llm, simple_tools, agent_prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=simple_tools,
                handle_parsing_errors=(
                "Formato incorrecto. Recuerda terminar exactamente así, sin usar Action para ello:\n"
                "Thought: Ya tengo suficiente información.\n"
                "Final Answer: [tu respuesta]"
                ),
                verbose=True,
                max_iterations=10,
            )

            mensajes = db.get_messages(st.session_state.session_id)
            if memory_enabled:
                chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in mensajes[-297:]])
            else:
                chat_history = ""
            conversation_summ = db.get_conversation_summary(st.session_state.session_id) or ""

            response = await executor.ainvoke({
                "input": user_input,
                "chat_history": chat_history,
                "user_profile": db.get_user_profile_text(st.session_state.user_id) if memory_enabled else "No disponible (memoria desactivada).",
                "conversation_summary": conversation_summ if memory_enabled else "",
                "session_id": str(st.session_state.session_id),
                "user_id": str(st.session_state.user_id),
                "emotions": emotions,
                "predicted_emotion": predicted_emotion or "No disponible"
            })
            return response["output"]
        finally:
           if hasattr(client, "close"):
            await client.close()
           elif hasattr(client, "aclose"):
            await client.aclose() 


async def run_reset_agent(user_input: str, mode: str) -> str:
    client = MultiServerMCPClient(server_location)
    mcp_tools = await client.get_tools()

    if mode == "perfil":
        RESET_TOOLS = {"edit_profile_value", "reset_user_profile_fields"}
    elif mode == "eventos":
        RESET_TOOLS = {"edit_event", "get_important_events"}
    else:  # resumen
        RESET_TOOLS = {"edit_session_summary"}

    simple_tools = []
    for mcp_tool in mcp_tools:
        if mcp_tool.name not in RESET_TOOLS:
            continue

        def make_wrapper(t):
            async def wrapper(text: str) -> str:
                try:
                    arg_data = json.loads(text) if "{" in text else {}
                    if t.name == "edit_profile_value":
                        result = await t.ainvoke({
                            "user_id": str(st.session_state.user_id),
                            "field": arg_data.get("field", ""),
                            "action": arg_data.get("action", ""),
                            "value": arg_data.get("value"),
                            "old_value": arg_data.get("old_value"),
                            "new_value": arg_data.get("new_value")
                        })
                    elif t.name == "reset_user_profile_fields":
                        result = await t.ainvoke({
                            "user_id": str(st.session_state.user_id),
                            "fields": arg_data.get("fields", "todo")
                        })
                    elif t.name == "edit_event":
                        result = await t.ainvoke({
                            "session_id": str(st.session_state.session_id),
                            "action": arg_data.get("action", ""),
                            "event_id": arg_data.get("event_id"),
                            "event": arg_data.get("event"),
                            "new_type": arg_data.get("new_type"),
                            "new_importance": arg_data.get("new_importance")
                        })
                    elif t.name == "get_important_events":
                        result = await t.ainvoke({
                            "session_id": str(st.session_state.session_id)
                        })
                    elif t.name == "edit_session_summary":
                        result = await t.ainvoke({
                            "session_id": str(st.session_state.session_id),
                            "action": arg_data.get("action", ""),
                            "value": arg_data.get("value"),
                            "old_value": arg_data.get("old_value"),
                            "new_value": arg_data.get("new_value")
                        })
                    else:
                        result = await t.ainvoke({"query": text})

                    if isinstance(result, list):
                        return result[0].get("text", str(result))
                    return str(result)

                except Exception as e:
                    return f"Error: {e}"
            return wrapper

        simple_tools.append(Tool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            coroutine=make_wrapper(mcp_tool),
            func=lambda x: x
        ))

    if mode == "perfil":
        prompt_text = PROMPT_GESTION_PERFIL
    elif mode == "eventos":
        prompt_text = PROMPT_GESTION_EVENTOS
    else:
        prompt_text = PROMPT_GESTION_RESUMEN

    prompt = PromptTemplate.from_template(prompt_text)
    llm = get_llm(st.session_state.reasoning_enabled)
    agent = create_react_agent(llm, simple_tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=simple_tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=5
    )

    response = await executor.ainvoke({
        "input": user_input,
        "user_id": str(st.session_state.user_id),
        "session_id": str(st.session_state.session_id),
        "agent_scratchpad": ""
    })
    return response["output"]

@st.dialog("Gestionar perfil")
def profile_dialog():
    st.caption("Dime qué quieres añadir, cambiar o eliminar de tu perfil.")
    user_input = st.chat_input("Ej: cambia mi objetivo de X a Y")
    if user_input:
        with st.spinner("Procesando..."):
            response = asyncio.run(run_reset_agent(user_input, mode="perfil"))
        st.markdown(response)
        st.rerun()

@st.dialog("Gestionar eventos")
def events_dialog():
    st.caption("Dime qué evento quieres añadir, cambiar o eliminar.")
    user_input = st.chat_input("Ej: cambia la importancia del evento del ascenso a media")
    if user_input:
        with st.spinner("Procesando..."):
            response = asyncio.run(run_reset_agent(user_input, mode="eventos"))
        st.markdown(response)
        st.rerun()


@st.dialog("Gestionar resumen")
def summary_dialog():
    st.caption("Dime qué quieres añadir, cambiar o eliminar del resumen de esta sesión.")
    user_input = st.chat_input("Ej: elimina la parte sobre el accidente laboral")
    if user_input:
        with st.spinner("Procesando..."):
            response = asyncio.run(run_reset_agent(user_input, mode="resumen"))
        st.markdown(response)
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
    help="Desactivarlo reduce el tiempo de respuesta pero puede afectar a la calidad del razonamiento."
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


    try:
        traduccion = translator(user_input)[0]['translation_text']
        resultado = depression_classifier(traduccion)[0]
        print(f"Traducción: {traduccion}")
        print(f"Resultado: {resultado}")
        if resultado['label'] == 'moderate':
            if tiene_mas_emociones_negativas(emotion_data, umbral=0.5)[0] or tiene_mas_emociones_negativas(emotion_data, umbral=0.5)[1]:
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