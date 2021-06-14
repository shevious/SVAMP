from src.prepare_infer import load_infer_data

infer_ls = load_infer_data('/home/agc2021/dataset/problemsheet.json')
#print(infer_ls)
answers = {}
for q in infer_ls:
    answers[q['id']] = {
        'answer': '0',
        'equation': 'print(0)',
    }

import json
json_path = 'answersheet.json'
with open(json_path, 'w', encoding='UTF-8') as fout:
    json.dump(answers, fout, ensure_ascii=False, indent=4)
