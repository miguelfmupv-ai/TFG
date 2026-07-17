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
    "Dado una id de sesión, obtiene todos los eventos relevantes ocurridos en ella, incluyendo su tipo e importancia"
    
    events = db.get_events(session_id)
    return str([
        {"id": e["id"], "event": e["event"], "type": e["type"], "importance": e["importance"], "date": str(e["date"])}
        for e in events
    ])



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
def edit_profile_value(
    user_id: str,
    field: str,
    action: str,
    value: Optional[str] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None
) -> str:
    """Gestiona un valor dentro de un campo del perfil del usuario.
    field: nombre, relaciones, objetivos, temas, aficiones.
    action: 'modify' (añade un valor nuevo si old_value está vacío, o sustituye old_value por new_value si se indica),
            'remove' (elimina un valor concreto usando 'value'),
            'clear' (borra el campo entero de golpe).
    """
    try:
        return db.edit_profile_value(user_id, field, action, value, old_value, new_value)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def edit_event(
    session_id: str,
    action: str,
    event_id: Optional[str] = None,
    event: Optional[str] = None,
    new_type: Optional[str] = None,
    new_importance: Optional[str] = None
) -> str:
    """Gestiona los eventos de la sesión.
    action: 'modify' (crea un evento nuevo si event_id está vacío, o modifica el evento existente si se indica event_id),
            'remove' (elimina un evento usando event_id),
            'clear' (elimina TODOS los eventos de la sesión de golpe).
    Usa get_important_events primero si necesitas el event_id para modificar o eliminar un evento concreto.
    """
    try:
        return db.edit_event(session_id, action, event_id, event, new_type, new_importance)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def edit_session_summary(
    session_id: str,
    action: str,
    value: Optional[str] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None
) -> str:
    """Gestiona el resumen de la sesión actual.
    action: 'modify' (añade un fragmento nuevo si old_value está vacío, o sustituye old_value por new_value si se indica),
            'remove' (elimina fragmentos que contengan 'value'),
            'clear' (borra todo el resumen de golpe).
    """
    try:
        return db.edit_session_summary(session_id, action, value, old_value, new_value)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def flag_crisis_status(user_id: str) -> str:
    """Marca al usuario actual con estado de crisis activo.
    Úsala SIEMPRE que detectes señales de riesgo de crisis en el mensaje del usuario
    (ideación suicida, autolesión, desesperanza grave, planificación de despedida),
    incluso si no estás completamente seguro. Ante la duda, márcalo.
    """
    try:
        db.set_crisis_status(user_id, True)
        return "Estado de crisis marcado."
    except Exception as e:
        return f"Error: {e}"
        
if __name__ == "__main__":
    mcp.run(transport="stdio")
