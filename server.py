from typing import Optional
from mcp.server.fastmcp import FastMCP
import database as db


mcp = FastMCP("CHAT_TFG")


@mcp.tool()
def save_important_event(session_id: str, event: str, new_type: str = "general", importance: str = "moderada") -> str:
    """Guarda un evento importante en la base de datos en texto plano. Usa la id de la sesión para relacionarlo con la conversación actual. El tipo y la importancia son opcionales pero pueden ayudar a categorizar el evento."""
    try:
        db.create_event(session_id, event, new_type, importance)
        return "Evento guardado."
    except Exception as e:
        return f"Error: {e}"



@mcp.tool()
def get_important_events(session_id: str) -> str:
    "Dado una id de sesión, obtiene todos los eventos relevantes ocurridos en ella"
    
    events = db.get_events(session_id)
    return str([{"id": e["id"], "event": e["event"], "date": str(e["date"])} for e in events])



@mcp.tool()
def conversation_briefer(session_id: str, summary: str) -> str:
    "Guarda el resumen muy breve de la conversación en la base de datos en formato texto plano. Ayúdate del evento que ha disparado esta herramienta para generar el resumen. Usa la id de la sesión para relacionarlo con la conversación actual. En el campo de resumen debes poner el resumen de la conversación"
    
    db.update_session(session_id, new_summary=summary)
    return "Resumen guardado."

@mcp.tool()
def update_user_profile(user_id: str, name: Optional[str] = None, relationships: Optional[str] = None, goals: Optional[str] = None, topics: Optional[str] = None, hobbies: Optional[str] = None) -> str:
    """Actualiza la ficha biográfica del usuario. 
    Llama a esta herramienta cuando el usuario te cuente su nombre o datos nuevos sobre sus relaciones, metas, aficiones o temas de interés.
    """
    try:
        db.update_profile(user_id, name, relationships, goals, topics, hobbies)
        return "Perfil actualizado."
    except Exception as e:
        return f"Error: {e}"
    
@mcp.tool()
def get_user_profile(user_id: str) -> str:
    """Recupera la ficha biográfica del usuario (relaciones, metas y temas recurrentes).
    Úsala si necesitas recordar quién es el usuario o verificar sus objetivos actuales.
    """
    try:
        profile_text = db.get_user_profile_text(user_id)
        return profile_text
    except Exception as e:
        return f"Error al recuperar el perfil: {e}"


@mcp.tool()
def delete_event(session_id: str, event_id: str) -> str:
    """Borra un evento específico de la sesión por su ID.
    Antes de borrar, deberás haber usado get_important_events para obtener los IDs de los eventos
    y así identificar cuál quiere eliminar el usuario.
    """
    try:
        db.delete_event_by_id(event_id)
        return "Evento eliminado."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def reset_user_profile_fields(user_id: str, fields: str) -> str:
    """Borra campos específicos del perfil del usuario cuando este pide explícitamente
    que olvides algo concreto sobre él.
    El parámetro fields acepta: todo, nombre, relaciones, objetivos, temas, aficiones.
    Ejemplo: si el usuario dice 'olvida mis objetivos', usa fields='objetivos'.
    """
    try:
        db.reset_profile_fields(user_id, fields)
        return "Información olvidada."
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
