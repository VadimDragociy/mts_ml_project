from openai import OpenAI

api_key = open("api_key.txt", 'r').read()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def validate_from_gemini(query, answer, answer_LLM):
    prompt = f"""
    Проанализируй вопрос и эталонный ответ с моим ответом, оцени правильность моего ответа по шкале от 1 до 100, где 1 - полное несовпадение, а 100 - полное совпадение
    Вопрос: {query}
    Эталонный ответ: {answer}
    Мой ответ: {answer_LLM}
    """
    # completion = client.chat.completions.create(
    #     # extra_headers={
    #     #   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    #     #   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    #     # },
    #     model="google/gemini-2.0-flash-001",
    #     messages=[
    #         {
    #         "role": "user",
    #         "content": prompt
    #         }
    #     ]
    #     )
    # TODO: убрать запись отсюда куда нибудь в адекватное место
    with open("output.txt", 'w') as _:
        print("answer: ", answer_LLM, '\n')
        _.write(f"""{answer_LLM}""")
        _.write('\n')
    with open("evaluation.txt", 'w') as _:
        # print("evaluation: ", completion.choices[0].message.content, '\n')
        # _.write(f"""{completion.choices[0].message.content}""")
        print("evaluated..............")
        _.write(f"""evaluated""")
        _.write('\n')
