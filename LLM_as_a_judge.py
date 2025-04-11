from openai import OpenAI

api_key = open("/api_key", 'r').read()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

# completion = client.chat.completions.create(
#   # extra_headers={
#   #   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#   #   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   # },
#   model="Gemini 2.0 Flash",
#   messages=[
#     {
#       "role": "user",
#       "content": "What is the meaning of life?"
#     }
#   ]
# )

def validate_from_gemini(query, answer, answer_LLM):
  prompt = f"""
  Проанализируй вопрос и эталонный ответ с моим ответом, оцени правильность моего ответа по шкале от 1 до 100, где 1 - полное несовпадение, а 100 - полное совпадение
  Вопрос: {query}
  Эталонный ответ: {answer}
  Мой ответ: {answer_LLM}
  """
  completion = client.chat.completions.create(
  # extra_headers={
  #   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
  #   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  # },
    model="Gemini 2.0 Flash",
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ]
  )
  with open("/evaluation.txt") as _:
    _.write(completion.choices[0].message.content)
    _.write('/n')
