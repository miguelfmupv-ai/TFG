


Requisitos previos

Python 3.10 o superior
Ollama instalado y corriendo en local → ollama.com. Una vez instalado, descarga el modelo que vayas a usar (por defecto el script usa gemma4:e4b):

  ollama pull gemma4:e4b

CUDA 12.1 si quieres aprovechar la GPU. El requirements.txt incluye torch==2.5.1+cu121; si no tienes GPU o usas una versión distinta de CUDA, cambia esa línea antes de instalar:

  torch==2.5.1        # solo CPU
  torch==2.5.1+cu118  # CUDA 11.8

Conexión a internet en el primer arranque: el modelo de detección de emociones (tabularisai/multilingual-emotion-classification) se descarga automáticamente de Hugging Face.


Cómo funciona la app
Al ejecutar streamlit run chat.py se abre una interfaz de chat en el navegador. En el panel lateral puedes gestionar tus conversaciones y activar o desactivar tres funcionalidades de forma independiente:

🧠 Detector de emociones — analiza cada mensaje que envías e identifica las emociones presentes (alegría, tristeza, miedo, rabia, etc.) con su nivel de confianza. El resultado se muestra encima de la respuesta del asistente.
🗃️ Memoria — el asistente recuerda información que le cuentes sobre ti: tu nombre, relaciones, objetivos, aficiones y temas recurrentes. Esta información se guarda en una base de datos local y persiste entre sesiones.
📝 Resumen de sesión — cuando ocurre un evento relevante (un ascenso, un viaje, una ruptura...), el asistente guarda automáticamente un resumen de lo ocurrido para poder recordarlo más adelante.

El asistente también dispone de un sistema de detección de crisis: si combina una emoción negativa dominante con ciertas expresiones en el mensaje, bloquea el chat y muestra recursos de ayuda profesional (línea 024 y emergencias 112).
La base de datos (TFG_CHAT.db) y el servidor interno de herramientas (server.py) se gestionan automáticamente; no es necesario configurarlos ni arrancarlos manualmente.










Recuerda descargarte el modelo que vayas a usar en Ollama en tu propio entorno y configurarlo en el script "chat.py".

Para ejecutarlo, coloca los tres archivos "chat.py", "server.py", "database.py" en la misma ubicación y ejecuta el siguiente comando por consola:

>streamlit run chat.py

Si quieres comprobar el estado de la base de datos ejecuta el siguiente script:

>python prueba.py

Si quieres usar el requirements.txt directamente usa esto:
pip install -r requirements.txt --break-system-packages

Si quieres ejecutar el pip install usa esto:
pip install datasets==5.0.0 langchain_classic==1.0.8 langchain_core==1.4.6 langchain_mcp_adapters==0.3.0 langchain_ollama==1.1.0 matplotlib==3.11.0 mcp==1.27.2 nest_asyncio==1.6.0 numpy==2.4.6 pandas==3.0.3 scikit_learn==1.9.0 SQLAlchemy==2.0.50 streamlit==1.56.0 tabulate==0.10.0 torch==2.5.1+cu121 tqdm==4.67.1 transformers==4.40.0 --break-system-packages