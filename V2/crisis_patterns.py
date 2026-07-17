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
    r"quiero\s+desaparecer",
    r"quitarme\s+la\s+vida",
    r"acabar\s+con\s+(todo|mi\s+vida|esto)",
    r"quiero\s+morir",
    r"dormirme\s+y\s+no\s+volver\s+a\s+despertar",
    r"mejor\s+si\s+yo\s+no\s+existiera",
    r"(el\s+mundo|todos)\s+estar[ií]a(n)?\s+mejor\s+sin\s+m[ií]",
    r"nunca\s+voy\s+a\s+mejorar",
    r"elegir[ií]a\s+no\s+haber\s+nacido",
    r"no\s+hay\s+manana\s+para\s+mi",
    r"no\s+me\s+veo\s+en\s+el\s+futuro",
    r"ya\s+no\s+aguanto\s+mas\s+esta\s+situacion",
    r"no\s+se\s+cuanto\s+tiempo\s+mas\s+voy\s+a\s+poder\s+resistir",
    r"soy\s+una\s+carga\s+para",
    r"nadie\s+notaria\s+(demasiado\s+)?si",
    r"dejara\s+de\s+aparecer.*nadie\s+lo\s+notaria",
    r"ya\s+no\s+import[oa]",
    r"estaria(n)?\s+mas\s+tranquil[oa]s?\s+si\s+yo\s+no",
    r"no\s+voy\s+a\s+estar\s+(aqui\s+)?manana",
    r"cuidate.*cuando\s+ya\s+no\s+este",
    r"deciros\s+adios",
    r"me\s+he\s+despedido\s+ya\s+de",
    r"despidiendome\s+del\s+mundo",
    r"me\s+rindo,",
    r"esta\s+bien\s+rendirse",
    r"se\s+como\s+terminar\s+con\s+todo",
    r"ya\s+he\s+tomado\s+la\s+decision",
    r"planeando\s+como\s+desaparecer",
    r"decidido\s+irme\s+para\s+siempre",
    r"ordenando\s+mis\s+cosas.*(por\s+si\s+acaso|ya\s+no\s+este)",
    r"regalado\s+mis\s+cosas\s+mas\s+importantes.*ya\s+no\s+las\s+necesito",
    r"cancelado\s+todos\s+mis\s+planes\s+futuros",
    r"pedido\s+perdon\s+a\s+todo\s+el\s+mundo.*dejarlo\s+todo\s+arreglado",
    r"carta.*(por\s+que\s+he\s+decidido\s+irme|explicando)",
    r"suicidar(?:me|se)\b",
    r"cortar(?:me)?\s+las\s+venas\b",
    r"ahorcar(?:me)?\b",
    r"sobredosis",
    r"metodos\s+para\s+morir",
    r"formas\s+de\s+hacerlo.{0,30}(?:rapido|sin\s+dolor)",
]

# Anti-trampas: si aparecen, NO disparar aunque coincida un patrón crítico
# (para no convertir el regex en generador de falsos positivos)
PATRONES_EXCLUSION = [
    r"\b(pelicula|peli|serie|libro|novela|personaje|protagonista)\b",
    r"\bmi\s+(primo|prima|hermano|hermana|amigo|amiga|vecino|vecina)\b.*\b(paso|llego|penso|pensaba)\b",
    r"\bpor\s+suerte\b",
    r"\bal\s+final\s+(he|hemos|consegui)\b",
    r"\bantes\s+pensaba\b",
    r"\bpidio\s+ayuda\b",
    r"\b(dieta|trafico|calor|compra|presentacion|jefe)\b",
]

def detectar_riesgo_regex(texto: str) -> dict:
    t = _normalizar(texto)

    if any(re.search(p, t) for p in PATRONES_EXCLUSION):
        return {"match_critico": False, "patron": None, "motivo": "excluido por contexto"}

    for patron in PATRONES_CRITICOS:
        if re.search(patron, t):
            return {"match_critico": True, "patron": patron, "motivo": f"regex crítico: {patron}"}

    return {"match_critico": False, "patron": None, "motivo": None}