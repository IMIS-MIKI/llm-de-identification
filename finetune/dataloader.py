from typing import Literal, List
from unsloth.chat_templates import get_chat_template
from datasets import Dataset

import os
import json
import hashlib
import runner.src.prompts as prompts

RAW_DIR = 'raw'
DATA_DIR = 'data'


def load(datasets: List[Literal["uksh-ris", "uksh-anamnese", "2024-043"]] | None = None):
    """Load dataset available for finetuning of the model. These correspond to initial ris+anamnese data request and
    future data anonymized. This data needs to be manually checked and curated to be eligible for fine-tuning.
    Afterward, just add the new batch to the list and how to fetch it.
    Final data should be a dictionary where each entry is {'processed': anonimized-text, 'text': original text}
    """

    if datasets is None:
        datasets = ["uksh-ris", "uksh-anamnese", "2024-043"]

    results = dict()

    # TODO:
    # Necessary to update results to add the content to train the model with
    # Structure should be:
    # results[key] = {'processed': 'de-identified text comes here', 'text': 'Original text comes her'}

    return results


def clean(txt):
    """Ursprüngliche clean Funktion"""
    txt = txt.replace('{', '')
    txt = txt.replace(' "text": "', '')
    txt = txt.replace('}', '')
    txt = txt.replace('\\n', ' ')
    txt = txt.replace('\n', ' ')
    txt = txt[:-2]
    txt = txt.lstrip()
    txt = ' '.join(txt.split())
    return txt


def get(directory, prefix, path):
    """Ursprüngliche load Funktion"""
    d = json.load(open(directory + '/' + prefix + '/' + path))
    if directory == RAW_DIR:
        return clean(d['text'])
    else:
        return clean(d['masked'])


def convert_to_gemma3_format(data_dict):
    """Konvertiert Daten in das Gemma 3 Chat-Format"""

    formatted_data = []

    for filename, content in data_dict.items():
        original_text = content['text']
        processed_text = content['processed']

        # Qualitätsprüfung
        if len(original_text) < 50 or len(processed_text) < 50:
            print(f"Skipping {filename}: Text zu kurz")
            continue

        if len(original_text) > 6000 or len(processed_text) > 6000:
            print(f"Warning {filename}: Text sehr lang ({len(original_text)}/{len(processed_text)} chars)")
            # Für sehr lange Texte könnten wir sie aufteilen

        user_prompt = prompts.user_prompt(original_text)

        # Gemma 3 Format: System-Prompt wird in User-Message eingebettet
        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": processed_text}
        ]

        formatted_data.append({"conversations": conversation})

    print(f"✅ Converted {len(formatted_data)} conversations to Gemma 3 format")
    return formatted_data


def create_gemma_dataset(formatted_data, tokenizer):
    """Erstellt ein Dataset für das Training mit Gemma 3 Template"""
    try:
        print("Creating training dataset with Gemma 3 template...")

        # Wende das Gemma 3 Chat Template an
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",  # Gemma 3 spezifisches Template
        )

        def formatting_prompts_func(examples):
            """Formatiert die Prompts für das Training"""
            convos = examples["conversations"]
            texts = []
            for convo in convos:
                text = tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)
            return {"text": texts}

        # Erstelle Dataset
        dataset = Dataset.from_list(formatted_data)
        dataset = dataset.map(formatting_prompts_func, batched=True)

        print(f"✅ Dataset created with {len(dataset)} examples")
        print(f"Sample text length: {len(dataset[0]['text'])} characters")

        # Zeige ein Beispiel der Formatierung
        print("\n📝 Sample formatted text:")
        print("=" * 80)
        print(dataset[0]['text'][:500] + "...")
        print("=" * 80)

        return dataset

    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return None


# Just for testing - actual run in finetune_gemma3_27b
if __name__ == '__main__':
    data = load()
    conversations = convert_to_gemma3_format(data)
    # create_gemma_dataset(conversations)
    print(conversations)
