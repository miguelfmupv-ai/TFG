"""
crisis_patterns.py
===================
Patrones regex de alto riesgo, usados como red de seguridad
independiente del pipeline de ML en evaluar_riesgo_crisis (utils.py).
"""
import re
import unicodedata

def _normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")  # quita tildes
    return texto

# Nivel crítico: dispara crisis=True siempre, sin pasar por el modelo
PATRONES_CRITICOS = [
    r"quitarme\s+la\s+vida",
    r"no\s+quiero\s+seguir\s+viviendo",
    r"acabar\s+con\s+(mi\s+vida|todo|esto)",
    r"hacerme\s+dano",
    r"no\s+aguanto\s+mas",
    r"no\s+puedo\s+mas",
    r"mejor\s+(estaria|estar)\s+muert[oa]",
    r"quiero\s+morir",
    # completar con frases reales extraídas de tus FN del dataset
]

# Anti-trampas: si aparecen, NO disparar aunque coincida un patrón crítico
# (para no convertir el regex en generador de falsos positivos)
PATRONES_EXCLUSION = [
    r"no\s+quiero\s+morir",
    r"pel[ií]cula|libro|canci[oó]n|personaje",  # referencias ficticias/contexto ajeno
]

def detectar_riesgo_regex(texto: str) -> dict:
    t = _normalizar(texto)

    if any(re.search(p, t) for p in PATRONES_EXCLUSION):
        return {"match_critico": False, "patron": None, "motivo": "excluido por contexto"}

    for patron in PATRONES_CRITICOS:
        if re.search(patron, t):
            return {"match_critico": True, "patron": patron, "motivo": f"regex crítico: {patron}"}

    return {"match_critico": False, "patron": None, "motivo": None}