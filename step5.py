from google import genai

client = genai.Client()

respons = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Jelaskan apa itu demensia dalam satu kalimat singkat"
)

print(respons.text)