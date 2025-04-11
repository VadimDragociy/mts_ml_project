# from transformers import T5ForConditionalGeneration, T5Tokenizer
# MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
# model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
# tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
# # model.cuda()
# # model.eval()

# # def paraphrase_test(relevant_chunk, question):
# #     paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
# #     input_text = f"paraphrase: {relevant_chunk} question: {question}"
# #     output = paraphraser(input_text, max_length=128, do_sample=True, top_k=50, num_return_sequences=1)
# #     print(output[0]['generated_text'])

# def paraphrase(relevant_chunk, question, beams=5, grams=4):
#     input_text = f"paraphrase: {relevant_chunk} question: {question}"
#     x = tokenizer(input_text, return_tensors='pt', padding=True).to(model.device)
#     max_size = int(x.input_ids.shape[1] * 1.5 + 10)
#     out = model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size)
#     print(tokenizer.decode(out[0], skip_special_tokens=True))


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "MTSAIR/Cotype-Nano"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",
)

def ask_model(prompt, max_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def paraphrase(relevant_chunk, search_query):
    prompt = f"""
        Далее отвечай по шаблону
        Вопрос: {search_query}
        Ответь, на вопрос, используя этот абзац: {relevant_chunk}
        ### Ответ:
    """
    answer = ask_model(prompt)
    return answer