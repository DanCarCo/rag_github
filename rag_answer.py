import os
from dotenv import load_dotenv
from openai import OpenAI

# Cargar token desde .env
load_dotenv()
token = os.getenv("GITHUB_TOKEN")

# Configuración del cliente OpenAI para GitHub
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def generate_answer(context, question):
    prompt = f"""Contexto:\n{context}\n\nPregunta:\n{question}\n\nRespuesta:"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Eres un asistente útil y preciso."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=1,
        model=model
    )

    return response.choices[0].message.content.strip()