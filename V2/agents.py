"""
agents.py
=========
Módulo que encapsula toda la lógica de orquestación de agentes (LangChain
ReAct + herramientas MCP) del sistema.

Expone:
    - `get_llm(reasoning)`: fábrica del modelo de lenguaje (Ollama/Gemma).
    - `run_agent(...)`: agente conversacional principal, con acceso a las
      herramientas de memoria/eventos/resumen según los toggles activos.
    - `run_reset_agent(...)`: agente especializado en operaciones de edición
      directa (gestionar perfil, eventos o resumen desde los diálogos de la UI).

La construcción de prompts se delega en `prompts.py`; este módulo se centra
en la instanciación de modelos, el envoltorio (wrapping) de herramientas MCP
como `Tool` de LangChain, y la ejecución de los `AgentExecutor`.
"""

import json

import streamlit as st
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool

import database as db
from prompts import (
    build_prompt,
    PROMPT_GESTION_PERFIL,
    PROMPT_GESTION_EVENTOS,
    PROMPT_GESTION_RESUMEN,
)

server_location = {
    "chat_tfg": {
        "command": "python",
        "args": ["server.py"],
        "transport": "stdio"
    }
}

def get_llm(reasoning: bool):
    return ChatOllama(
        model="gemma4:e4b",
        num_gpu=99,
        num_ctx=8192,
        reasoning=reasoning
    )

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
                            elif t.name == "flag_crisis_status":
                                result = await t.ainvoke({
                                    "user_id": str(st.session_state.user_id)
                                })
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


            EDIT_TOOLS = {"edit_profile_value", "edit_event", "edit_session_summary",
                          "reset_user_profile_fields", "delete_event"}
            simple_tools = [t for t in simple_tools if t.name not in EDIT_TOOLS]

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
                chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in mensajes[-90:]])
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
                "predicted_emotion": predicted_emotion or "No disponible",
                "agent_scratchpad": ""
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
    estado_anterior_razonamiento = st.session_state.reasoning_enabled
    st.session_state.reasoning_enabled = False  # Desactivar razonamiento extendido para estas operaciones
    llm = get_llm(st.session_state.reasoning_enabled )
    agent = create_react_agent(llm, simple_tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=simple_tools,
        handle_parsing_errors=True,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5
    )

    result = await executor.ainvoke({
        "input": user_input,
        "user_id": str(st.session_state.user_id),
        "session_id": str(st.session_state.session_id),
        "agent_scratchpad": ""
    })
    response = result["output"]
    steps = result.get("intermediate_steps", [])

    EDIT_TOOLS = {"edit_profile_value", "edit_event", "edit_session_summary", "reset_user_profile_fields"}

    edit_steps = [
        (action, observation)
        for action, observation in steps
        if action.tool in EDIT_TOOLS
    ]

    st.session_state.pending_toasts = [
        f"✅{observation}"
        for action, observation in edit_steps
    ]
    st.session_state.reasoning_enabled = estado_anterior_razonamiento
    return response
