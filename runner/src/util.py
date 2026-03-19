from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import os
import time
import runner.src.prompts as prompts

load_dotenv()


def convert_to_viewer_format(text, processed, runtime):
    res = dict()
    res['text'] = text
    res['result'] = processed
    if runtime:
        res['duration'] = runtime
    else:
        res['duration'] = 1.0
    res['is_valid'] = False
    return res


def clean(txt):
    """Ursprüngliche clean Funktion"""
    txt = txt.replace('{', '')
    txt = txt.replace(' "text": "', '')
    txt = txt.replace('}', '')
    txt = txt.lstrip()
    txt = ' '.join(txt.split())
    txt = txt.replace('\\y:575\\', '')
    txt = txt.replace('\\f:12052\\', '')
    txt = txt.replace('\\a:575\\', '')
    txt = txt.replace('\\y:551\\', '')
    txt = txt.replace('\\m0', '')
    txt = txt.replace('\\m1', '')
    txt = txt.replace('\\m8', '')
    return txt


def get_llm(model=os.getenv('DEFAULT_MODEL'), penalty=os.getenv('BASE_PENALTY')):
    ollama_host = os.getenv('OLLAMA_HOST')
    ollama_keep_alive = os.getenv('OLLAMA_KEEP_ALIVE')
    llm = OllamaLLM(temperature=0, top_k=20, top_p=0.3, base_url=ollama_host, model=model,
                    keep_alive=ollama_keep_alive, repeat_penalty=penalty)
    return llm


def run_text(text, penalty_run: False):
    if penalty_run:
        llm = get_llm(penalty=os.getenv('RETRY_PENALTY'))
        messages = [
            {
                "role": "system",
                "content": prompts.system_prompt_violation()
            },
            {
                "role": "user",
                "content": prompts.user_prompt(text)
            }
        ]
    else:
        llm = get_llm()
        messages = [
            {
                "role": "system",
                "content": prompts.system_prompt()
            },
            {
                "role": "user",
                "content": prompts.user_prompt(text)
            }
        ]

    start_time = time.time()
    result = llm.invoke(messages)
    end_time = time.time()
    runtime = end_time - start_time
    return result, runtime


def process_text(text):
    text = clean(text)
    if not check_whitelist(text):
        res, rt = run_text(text, penalty_run=False)

        if len(res) >= len(text) * 1.05:
            print('Hit penalty')
            res, rt = run_text(text, penalty_run=True)

        return res, rt
    else:
        return text, 0


whitelist = [
    "s.u.",
    "s.o.",
    "s.oben",
    "s.unten",
    "s. oben",
    "s. unten",
    "siehe oben",
    "siehe unten",
    "Normalbefund.",
    "wird nicht befundet",
    "Unauffälliger Befund.",
    "Befund folgt.",
    "wird nachgereicht",
    "BI-RADS 2 beidseits.",
    "Beidseits BI-RADS 2.",
    "Beidseits BI-RADS 2. ",
    "Keine zervikalen Traumafolgen.",
    "Siehe Befund Mammographie vom selben Datum.",
    ""
    "Keine Traumafolgen."
]


def check_whitelist(text):
    return text in whitelist
