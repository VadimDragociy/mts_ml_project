# from segmenter import Segmenter
from relevant import *
import json
from LLM_as_a_judge import *

if __name__ == '__main__':
    with open("dataSet/texts/deepseek_tale.txt", 'r', encoding='utf-8') as tale:
        string_tale = tale.read()
    # with open("dataSet/message_text.txt", 'r', encoding='utf-8') as metro:
    #     string_metro = metro.read()
    # with open("dataSet/metro_homecoming.txt", 'r', encoding='utf-8') as alg:
    #     string_alg = alg.read()

    with open('dataSet/query/query_tale.json', 'r', encoding='utf-8') as tale_query:
        query_tale = json.load(tale_query)

    # query_metro = 'сложные загадки'
    # query_metro_2 = 'подземелья'
    # query_alg = 'алгоритмы сортировки'
    # query_metro = 'Бумажная карта'
    # query_metro_2 = 'персонажи'
    # query_alg_1 = 'табуретка'

    # rlv_tale = Relevance(string_tale)
    # rlv_metro = Relevance(string_metro)
    # rlv_alg = Relevance(string_alg)

    # rlv_metro.search_for_query(query_metro_2)
    # rlv_metro.search_for_query(query_metro)
    # rlv_alg.search_for_query(query_alg)
    # rlv_alg.search_for_query(query_alg_1)
    rlv = Relevance_Frida(string_tale)
    for query in query_tale:
        print("------------------------------------------------------")

        # rlv_tale.search_for_query(query['question'])
        answers_frida = rlv.search_for_query(query['question'])
        for answer_frida in answers_frida:
            validate_from_gemini(query['question'], query['answer'], answer_frida)

        # print('Вопрос:', query['question'], 'Ответ:', query['answer'])
        print("------------------------------------------------------")
    #     rlv_tale.search_for_query(query)
    # rlv_tale.search_for_query(query_tale)
    # for query in query_tale:
    #     print(query)
