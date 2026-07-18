# Manual de usuario — Validador Emocional

Aplicación de apoyo emocional conversacional basada en un agente de IA (Gemma, vía Ollama) con memoria a largo plazo, detección de emociones y una capa de seguridad frente a situaciones de crisis.

---

## 1. Primeros pasos

Al abrir la aplicación por primera vez:

- Se crea automáticamente un **perfil de usuario** y una **primera sesión** ("Conversación actual"), sin necesidad de registro.
- La conversación se escribe en el cuadro de texto de la parte inferior de la pantalla.
- El nombre de la sesión se genera automáticamente a partir de tu primer mensaje (los primeros ~40 caracteres).

No necesitas configurar nada para empezar a hablar: basta con escribir tu primer mensaje.

---

## 2. La conversación

Cada vez que envías un mensaje:

1. El sistema analiza (opcionalmente) las **emociones** presentes en tu texto y las muestra como un conjunto de métricas (por ejemplo: 😢 Tristeza 62%, 😞 Decepción 41%...).
2. El sistema comprueba si el mensaje indica **riesgo de crisis**. Si es así, el chat se bloquea y se muestra un mensaje con recursos de ayuda (ver sección 6).
3. Si no hay riesgo, el agente conversacional genera una respuesta, que puede apoyarse en tu memoria guardada (perfil, eventos, resumen de la sesión) si esas funciones están activadas.

Cada respuesta tarda hasta 60 segundos; si se supera ese tiempo, verás un aviso pidiéndote que reformules el mensaje o lo intentes de nuevo.

Bajo cada uno de tus mensajes anteriores con emociones detectadas, se muestran las 5 emociones principales de ese mensaje como referencia.

---

## 3. La barra lateral

### 3.1. Nueva conversación

El botón **"✏️ Nueva conversación"** crea una sesión nueva. Al hacerlo, el sistema genera automáticamente un resumen acumulado de todo lo que sabe de ti (perfil + eventos + resúmenes de sesiones anteriores) y lo traslada como contexto inicial de la nueva sesión, para que no se pierda continuidad entre conversaciones.

### 3.2. Interruptores (toggles) disponibles

| Interruptor | Qué hace |
|---|---|
| 🧠 **Detector de emociones** | Activa/desactiva el análisis emocional de tus mensajes y la comprobación de riesgo de crisis basada en emociones. |
| 🔮 **Predicción emocional** | Predice tu posible estado emocional a partir de sesiones anteriores. Solo disponible si el detector de emociones está activo. |
| 🗃️ **Memoria** | Activa/desactiva que el agente guarde y recupere información sobre ti (perfil, eventos). Si la desactivas, también se desactiva automáticamente el resumen de sesión. |
| 📝 **Resumen de sesión** | El agente guarda un resumen breve al detectar hitos importantes en la conversación. Requiere que la memoria esté activada. |
| 🧩 **Razonamiento extendido** | Modo de razonamiento más profundo del modelo. Desactivarlo agiliza las respuestas, pero puede afectar a la calidad de algunas decisiones del agente. |

> **Nota:** aunque desactives todos los interruptores anteriores, la comprobación de seguridad ante situaciones de crisis permanece siempre activa como salvaguarda; no es configurable por diseño.

### 3.3. "Lo que sé de ti"

Muestra un resumen de tu perfil biográfico tal como lo tiene guardado el sistema (nombre, relaciones, objetivos, temas recurrentes, aficiones). Si todavía no hay información, se indica explícitamente.

**Gestionar perfil** abre un cuadro de diálogo donde puedes pedir en lenguaje natural que se añada, modifique o elimine cualquier dato del perfil. Ejemplos que puedes escribir tal cual:
- *"añade como afición que me gusta la fotografía"*
- *"cambia mi tema recurrente 'Amigos' por 'Ruptura'"*
- *"borra mi objetivo de aprender inglés"*
- *"borra todos mis temas"*

### 3.4. Sesión actual

Muestra el nombre de la sesión activa. El icono 🗑 permite eliminarla; al hacerlo, se cambia automáticamente a otra sesión existente o se crea una nueva si no queda ninguna.

### 3.5. Resumen de sesión

Muestra el resumen guardado de la conversación actual (si existe). El botón **"Gestionar resumen"** abre un diálogo para editarlo en lenguaje natural, con el mismo funcionamiento que el diálogo de perfil (añadir, modificar, eliminar o borrar todo el resumen).

### 3.6. Eventos de la sesión

Lista los hitos importantes que el agente ha detectado y guardado durante la conversación (con su tipo e importancia). El botón **"Gestionar eventos"** permite añadir, modificar o eliminar eventos concretos, o borrarlos todos, igual que en los diálogos anteriores.

### 3.7. Otras sesiones

Debajo aparece la lista del resto de tus conversaciones guardadas. Puedes cambiar a cualquiera de ellas pulsando su nombre, o eliminarlas con el icono 🗑.

### 3.8. Recargar la sesión

Vuelve a cargar la aplicación desde cero sin perder tu conversación ni tus datos guardados. Útil si la interfaz se queda bloqueada o quieres forzar una actualización visual.

### 3.9. Salir y cerrar aplicación

Cierra por completo la aplicación (no solo la ventana del navegador). Pide confirmación antes de ejecutarse, para evitar cierres accidentales.

---

## 4. Emociones detectadas

Cuando el detector de emociones está activo, tras cada mensaje tuyo verás:

- Un bloque **"🔍 Emociones detectadas"** con las 5 emociones más relevantes de tu mensaje y su nivel de confianza (en porcentaje).
- Estos porcentajes son independientes entre sí (no sumas 100%): cada emoción se evalúa por separado.
- Si la predicción emocional está activa, también verás un aviso **"🔮 Emoción predicha para esta sesión"**, con las 3 emociones que el sistema anticipa según el histórico de la conversación.

---

## 5. Memoria del sistema

El sistema puede recordar tres tipos de información entre conversaciones (si la memoria está activada):

- **Perfil biográfico**: nombre, relaciones, objetivos, temas recurrentes y aficiones.
- **Eventos importantes**: hitos puntuales de una sesión (por ejemplo, un logro laboral o una mala noticia).
- **Resumen de sesión**: un resumen breve y actualizado de cada conversación.

Toda esta información se usa como contexto para que el agente pueda responderte de forma más personalizada y coherente con conversaciones anteriores, y puedes revisarla o corregirla en cualquier momento desde los diálogos de gestión de la barra lateral.

---

## 6. Alerta de crisis

Si en algún momento el sistema detecta señales de riesgo grave (ideación suicida, autolesión, desesperanza extrema, etc.), la conversación se bloquea de inmediato y se muestra un mensaje con recursos de ayuda:

- 📞 **Línea de Prevención del Suicidio:** 024 (gratuita)
- 🚑 **Servicios de Emergencia:** 112
- 🫂 Recomendación de contactar con alguien de confianza

Mientras esta alerta esté activa, el cuadro de texto queda deshabilitado y no se pueden enviar más mensajes. Esta es una medida de seguridad y **no es un error del sistema**: es intencional y no puede desactivarse desde la interfaz.

> Este sistema es una herramienta de apoyo conversacional y **no sustituye la atención de un profesional de salud mental**. Ante cualquier indicio de crisis, prioriza siempre contactar con los servicios de emergencia o líneas de ayuda especializadas.

---

## 7. Solución de problemas

| Situación | Qué significa | Qué hacer |
|---|---|---|
| "Estoy tardando más de lo normal en responder..." | El agente ha superado el tiempo máximo de espera (60s). | Reformula el mensaje o inténtalo de nuevo. |
| "Parece que no puedo conectar con el modelo..." | Ollama no está activo o no responde. | Comprueba que el servicio de Ollama esté en marcha e inténtalo de nuevo. |
| "He tenido un problema procesando tu mensaje..." | El agente no ha podido completar su razonamiento a tiempo. | Intenta reformular con otras palabras. |
| "Ha ocurrido un error inesperado..." | Fallo no controlado. | Si persiste, usa "🔄 Recargar la sesión". |

---

*Este manual describe el comportamiento de la interfaz tal como está implementado en `chat.py`. Si el sistema evoluciona, actualiza este documento en consecuencia.*
