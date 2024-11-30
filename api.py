import os
from google.cloud import dialogflow
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import uvicorn

app = FastAPI()

# Configurar las credenciales
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

class Message(BaseModel):
    message: str

def detect_intent_texts(project_id, session_id, texts, language_code):
    print("Detectando intención...")
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)
    
    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)
        
        response = session_client.detect_intent(request={"session": session, "query_input": query_input})
        
        return response.query_result

@app.post("/webhook")
async def webhook(msg: Message):
    mensaje_recibido = msg.message
    
    # Configuración del proyecto y sesión
    project_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    session_id = str(uuid.uuid4())
    language_code = "es"
    
    result = detect_intent_texts(project_id, session_id, [mensaje_recibido], language_code)
    
    # Obtener la intención detectada y la respuesta de Dialogflow
    intencion_detectada = result.intent.display_name
    respuesta = result.fulfillment_text

    # Verificar si el contexto es `informaciongeneral-followup`
    for context in result.output_contexts:
        if 'informaciongeneral-followup' in context.name:
            # Verificar si la intención es una respuesta negativa
            if intencion_detectada == "informaciongeneral-no":
                respuesta = "¡Adiós! Que tengas un buen día."
                break
    
    return {"intencion_detectada": intencion_detectada, "respuesta": respuesta}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
