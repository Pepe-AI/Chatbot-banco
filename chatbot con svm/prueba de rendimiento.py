import pickle
from fastapi import FastAPI, Request
import uvicorn
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import spacy
import nltk
from nltk.corpus import stopwords
import string
import unicodedata
from uuid import uuid4

app = FastAPI()

# Descargar stopwords de nltk
nltk.download('stopwords')

# Cargar el modelo de Spacy para español
nlp = spacy.load('es_core_news_sm')

# Obtener la lista de stopwords en español y eliminar los interrogativos
stop_words = set(stopwords.words('spanish'))

# Lista de puntuación con caracteres en español
punctuation = string.punctuation

# Cargar las palabras del archivo diccionario_general.txt
with open('diccionario_general.txt', 'r', encoding='utf-8') as f:
    custom_stop_words = set(f.read().splitlines())

# Combinar las stopwords y las palabras personalizadas
stop_words.update(custom_stop_words)

# Diccionario para almacenar el estado de las sesiones
session_states = {}

# Función de preprocesamiento
def preprocess(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    text = text.lower()
    text = text.translate(str.maketrans('', '', punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words and token.lemma_ not in punctuation]
    return ' '.join(tokens)

def clean_text(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def contains_keyword(text, keywords):
    for keyword in keywords:
        if keyword in text:
            return keyword
    return None

try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading TF-IDF vectorizer: {e}")

try:
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
except Exception as e:
    print(f"Error loading SVM model: {e}")

label_map = {0: "consultar_saldo", 1: "cancelar_tarjeta", 2: "info_prestamo", 3: "info_seguros", 4: "info_tarjetas", 5: "info_general"}

def classify_message(message, threshold=0.8):
    try:
        message = preprocess(message)
        X = vectorizer.transform([message])
        probabilities = svm_model.predict_proba(X)[0]
        max_prob = max(probabilities)
        if max_prob < threshold:
            return "desconocido"
        label_idx = svm_model.predict(X)[0]
        return label_map.get(label_idx, "desconocido")
    except Exception as e:
        print(f"Error in classify_message: {e}")
        return "desconocido"

def extract_digits(message, length):
    digits = re.findall(r'\b\d{' + str(length) + r'}\b', message)
    return digits[0] if digits else None

def handle_data_request(session_id, message, intent):
    state = session_states[session_id]
    cleaned_message = clean_text(message)
    response_message = "No entendí su solicitud, por favor intente nuevamente."

    if state['waiting_for_digits']:
        card_number = extract_digits(message, 16)
        if card_number:
            state['collected_data']['card_number'] = card_number
            state['waiting_for_digits'] = False
            state['waiting_for_security_code'] = True
            if intent == "consultar_saldo":
                response_message = "Proporcione su Número de seguridad (CVV) de 6 dígitos para continuar."
            elif intent == "cancelar_tarjeta":
                response_message = "Proporcione su Número de seguridad (CVV) de 6 dígitos para proceder con la cancelación."
        else:
            response_message = "Error, verifique que su Número de tarjeta tenga 16 dígitos sin espacios."

    elif state['waiting_for_security_code']:
        security_code = extract_digits(message, 6)
        if security_code:
            state['collected_data']['security_code'] = security_code
            response_message = "¿Desea hacer otra consulta?"
            state['waiting_for_security_code'] = False
            state['waiting_for_final_confirmation'] = True
        else:
            response_message = "Error, verifique que su CVV tenga 6 dígitos sin espacios."

    elif state['waiting_for_loan_type']:
        loan_type = contains_keyword(cleaned_message, ["personal", "medico", "escolar", "pyme"])
        if loan_type:
            state['collected_data']['loan_type'] = loan_type
            response_message = f"Para solicitar un préstamo {loan_type}, los requisitos son: a, b, c. ¿Desea hacer otra consulta?"
            state['waiting_for_loan_type'] = False
            state['waiting_for_more_queries'] = True
        else:
            response_message = "Por favor, elija un tipo de préstamo disponible: Personal, Medico, Escolar, Pyme."

    elif state['waiting_for_insurance_type']:
        insurance_type = contains_keyword(cleaned_message, ["vida", "auto", "viajes"])
        if insurance_type:
            state['collected_data']['insurance_type'] = insurance_type
            response_message = f"Para obtener un seguro de {insurance_type}, los requisitos son: a, b, c. ¿Desea hacer otra consulta?"
            state['waiting_for_insurance_type'] = False
            state['waiting_for_more_queries'] = True
        else:
            response_message = "Por favor, elija un tipo de seguro disponible: vida, auto, viajes."

    elif state['waiting_for_card_type']:
        card_type = contains_keyword(cleaned_message, ["debito", "credito", "prepago"])
        if card_type:
            state['collected_data']['card_type'] = card_type
            response_message = f"Para obtener una tarjeta de {card_type}, los requisitos son: a, b, c. ¿Desea hacer otra consulta?"
            state['waiting_for_card_type'] = False
            state['waiting_for_more_queries'] = True
        else:
            response_message = "Por favor, elija un tipo de tarjeta disponible: débito, crédito, prepago."

    elif state['waiting_for_general_info']:
        general_info_type = contains_keyword(cleaned_message, ["tarjetas", "seguros", "prestamos"])
        if general_info_type:
            if general_info_type == "tarjetas":
                state['collected_data']['intent'] = "info_tarjetas"
                response_message = "Entendido, manejamos varios tipos de tarjetas: débito, crédito y prepago. ¿Sobre cuál te gustaría obtener más información?"
                state['waiting_for_card_type'] = True
            elif general_info_type == "seguros":
                state['collected_data']['intent'] = "info_seguros"
                response_message = "Entendido, en nuestro banco manejamos los siguientes seguros: vida, auto, viajes. ¿Sobre cuál te gustaría obtener más información?"
                state['waiting_for_insurance_type'] = True
            elif general_info_type == "prestamos":
                state['collected_data']['intent'] = "info_prestamo"
                response_message = 'Claro, manejamos los siguientes tipos de préstamos: Pyme, Personal, Escolar, Medico. ¿Sobre qué tipo de préstamo deseas información?'
                state['waiting_for_loan_type'] = True
            state['waiting_for_general_info'] = False
        else:
            response_message = "Por favor, elija un tipo de información disponible: tarjetas, seguros, préstamos."

    elif state['waiting_for_more_queries']:
        if message.lower() in ["sí", "si"]:
            response_message = "Entendido, ¿en qué más puedo ayudarte?"
            reset_state(session_id)
        elif message.lower() == "no":
            reset_state(session_id)
            if current_intent in ["info_prestamo", "info_seguros", "info_tarjetas"]:
                return {
                    "response_message": "¡Gracias por usar nuestro servicio! ¡Que tengas un buen día!",
                    "collected_data": {
                        "intent": "mensaje_final"
                    }
                }
            else:
                response_message = "¡Gracias por usar nuestro servicio! ¡Que tengas un buen día!"
                state['collected_data']['intent'] = "mensaje_final"
                return {
                    "response_message": response_message,
                    "collected_data": state['collected_data']
                }
        else:
            response_message = "Por favor, responda 'sí' para hacer otra consulta o 'no' para terminar."

    elif state['waiting_for_final_confirmation']:
        if message.lower() in ["sí", "si"]:
            response_message = "Entendido, ¿en qué más puedo ayudarte?"
            reset_state(session_id)
        elif message.lower() == "no":
            reset_state(session_id)
            return {
                "response_message": "¡Gracias por usar nuestro servicio! ¡Que tengas un buen día!",
                "collected_data": {
                    "intent": "mensaje_final"
                }
            }
        else:
            response_message = "Por favor, responda 'sí' para hacer otra consulta o 'no' para terminar."

    else:
        if intent == "consultar_saldo" or contains_keyword(cleaned_message, ["saldo", "dame mi saldo"]):
            state['collected_data']['intent'] = "consultar_saldo"
            response_message = "Para la consulta de saldo requerimos nos proporcione los 16 dígitos de su tarjeta."
            state['waiting_for_digits'] = True

        elif intent == "cancelar_tarjeta":
            state['collected_data']['intent'] = "cancelar_tarjeta"
            response_message = "Para cancelar su tarjeta requerimos nos proporcione los 16 dígitos de su tarjeta."
            state['waiting_for_digits'] = True

        elif intent == "info_prestamo":
            state['collected_data']['intent'] = "info_prestamo"
            response_message = 'Claro, manejamos los siguientes tipos de préstamos: Pyme, Personal, Escolar, Medico. ¿Sobre qué tipo de préstamo deseas información?'
            state['waiting_for_loan_type'] = True

        elif intent == "info_seguros":
            state['collected_data']['intent'] = "info_seguros"
            response_message = "Entendido, en nuestro banco manejamos los siguientes seguros: vida, auto, viajes. Si desea obtener información de cómo obtener uno, escriba el nombre del seguro que desee."
            state['waiting_for_insurance_type'] = True

        elif intent == "info_tarjetas":
            state['collected_data']['intent'] = "info_tarjetas"
            response_message = "Entendido, manejamos varios tipos de tarjetas: débito, crédito y prepago. ¿Sobre cuál te gustaría obtener más información?"
            state['waiting_for_card_type'] = True

        elif intent == "info_general" or contains_keyword(cleaned_message, ["informacion", "info", "informes"]):
            state['collected_data']['intent'] = "info_general"
            response_message = "Entendido, manejamos varios tipos de productos: tarjetas, seguros y préstamos. ¿Sobre cuál te gustaría obtener más información?"
            state['waiting_for_general_info'] = True

        else:
            response_message = f"Este bot puede ayudarte con las siguientes consultas: Consultar saldo, Cancelar tarjeta, Información de préstamos, Información de seguros, Información de tarjetas. ¿En qué puedo ayudarte?"

    session_states[session_id] = state
    return response_message

def reset_state(session_id):
    session_states[session_id] = {
        'waiting_for_digits': False,
        'waiting_for_security_code': False,
        'waiting_for_loan_type': False,
        'waiting_for_more_queries': False,
        'waiting_for_insurance_type': False,
        'waiting_for_card_type': False,
        'waiting_for_general_info': False,
        'waiting_for_final_confirmation': False,
        'collected_data': {},
        'current_intent': None
    }

@app.post("/message")
async def handle_message(request: Request):
    global session_states
    data = await request.json()
    user_message = data.get("message")
    session_id = data.get("session_id", str(uuid4()))  # Generar un nuevo UUID si no se proporciona un session_id

    if session_id not in session_states:
        reset_state(session_id)

    state = session_states[session_id]
    if any([state['waiting_for_digits'], state['waiting_for_security_code'], state['waiting_for_loan_type'], state['waiting_for_more_queries'], state['waiting_for_insurance_type'], state['waiting_for_card_type'], state['waiting_for_general_info'], state['waiting_for_final_confirmation']]):
        response = handle_data_request(session_id, user_message, state['current_intent'])
    else:
        intent = classify_message(user_message)
        state['current_intent'] = intent
        response = handle_data_request(session_id, user_message, intent)

    if isinstance(response, dict):  # Si la respuesta es un diccionario, significa que se está manejando la terminación de la conversación
        return response

    response_message = response

    api_response = {
        "response_message": response_message,
        "session_id": session_id
    }

    if 'card_number' in state['collected_data'] and 'security_code' in state['collected_data'] and state['current_intent'] in ["consultar_saldo", "cancelar_tarjeta"]:
        api_response["collected_data"] = {
            "intent": state['collected_data']['intent'],
            "card_number": state['collected_data']['card_number'],
            "security_code": state['collected_data']['security_code']
        }
        state['waiting_for_final_confirmation'] = True

    session_states[session_id] = state
    return api_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
