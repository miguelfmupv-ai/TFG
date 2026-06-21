

## Arquitectura

- `chat.py` — interfaz Streamlit y lógica principal
- `server.py` — servidor MCP con las herramientas del agente
- `database.py` — capa de persistencia con SQLAlchemy y SQLite

---

## Modelos utilizados

- **Gemma 4 E4B** (Ollama) — LLM principal del agente
- **AnasAlokla/multilingual_go_emotions** — detección de emociones multilingüe
- **Helsinki-NLP/opus-mt-es-en** — traducción español → inglés
- **rafalposwiata/deproberta-large-depression** — clasificador de riesgo de depresión

---

## Requisitos previos

- Python 3.10 o superior
- Ollama instalado y corriendo en local → ollama.com. Una vez instalado, descarga el modelo que vayas a usar (por defecto el script usa `gemma4:e4b`):
ollama pull gemma4:e4b
- CUDA 12.1 si quieres aprovechar la GPU. El `requirements.txt` incluye `torch==2.5.1+cu121`; si no tienes GPU o usas una versión distinta de CUDA, cambia esa línea antes de instalar:
torch==2.5.1        # solo CPU

torch==2.5.1+cu118  # CUDA 11.8
- Conexión a internet en el primer arranque: los modelos de detección de emociones y de depresión se descargan automáticamente de Hugging Face.

---

## Instalación

Usando `requirements.txt`:
pip install -r requirements.txt --break-system-packages

O instalando manualmente:
pip install datasets==5.0.0 langchain_classic==1.0.8 langchain_core==1.4.6 langchain_mcp_adapters==0.3.0 langchain_ollama==1.1.0 matplotlib==3.11.0 mcp==1.27.2 nest_asyncio==1.6.0 numpy==2.4.6 pandas==3.0.3 scikit_learn==1.9.0 SQLAlchemy==2.0.50 streamlit==1.56.0 tabulate==0.10.0 torch==2.5.1+cu121 tqdm==4.67.1 transformers==4.40.0 --break-system-packages
---

## Ejecución

Coloca los tres archivos `chat.py`, `server.py` y `database.py` en la misma carpeta y ejecuta:
streamlit run chat.py

Para comprobar el estado de la base de datos:
python prueba.py

La base de datos (`TFG_CHAT.db`) y el servidor interno de herramientas (`server.py`) se gestionan automáticamente; no es necesario configurarlos ni arrancarlos manualmente.

---

## Cómo funciona la app

Al ejecutar `streamlit run chat.py` se abre una interfaz de chat en el navegador. En el panel lateral puedes gestionar tus conversaciones y activar o desactivar las siguientes funcionalidades de forma independiente:

- 🧠 **Detector de emociones** — analiza cada mensaje que envías e identifica las emociones presentes (alegría, tristeza, miedo, rabia, etc.) con su nivel de confianza. El resultado se muestra encima de la respuesta del asistente.
- 🔮 **Predicción emocional** — predice la emoción dominante de la sesión basándose en el historial de sesiones anteriores.
- 🗃️ **Memoria** — el asistente recuerda información que le cuentes sobre ti: tu nombre, relaciones, objetivos, aficiones y temas recurrentes. Esta información se guarda en una base de datos local y persiste entre sesiones.
- 📝 **Resumen de sesión** — cuando ocurre un evento relevante (un ascenso, un viaje, una ruptura...), el asistente guarda automáticamente un resumen de lo ocurrido para poder recordarlo más adelante.
- 📌 **Eventos de sesión** — muestra los eventos importantes detectados en la conversación actual, con opción de borrarlos selectivamente mediante un chat con el agente.
- 👤 **Gestión de perfil** — permite indicarle al agente qué información biográfica debe olvidar mediante un chat directo.
- 🗑️ **Gestión de memoria** — desde el panel lateral puedes indicarle al agente, mediante un chat directo, qué datos biográficos o eventos de sesión quieres que olvide. El borrado es selectivo: puedes eliminar campos concretos del perfil (nombre, relaciones, objetivos, temas, aficiones) o eventos específicos de la sesión sin afectar al resto.
---


## Sistema de detección de crisis

El sistema opera en dos capas independientes:

1. Traducción del mensaje al inglés (`Helsinki-NLP/opus-mt-es-en`)
2. Clasificación de riesgo de depresión (`rafalposwiata/deproberta-large-depression`)

Si el clasificador detecta riesgo, el chat se bloquea y se muestran recursos de ayuda profesional (línea **024** y emergencias **112**).

---

## Limitaciones conocidas

- El clasificador de depresión puede generar falsos positivos en mensajes negativos cotidianos sin riesgo real (estrés laboral, cansancio, etc.).
- El detector de emociones clasifica mensajes de crisis con lenguaje indirecto o disociado como neutrales.
- El sistema está diseñado para español; el rendimiento en otros idiomas no ha sido evaluado.
- No sustituye atención psicológica profesional.
