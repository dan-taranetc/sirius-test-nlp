import config as cfg
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from tqdm.notebook import tqdm
import os
import re
import logging

tqdm.pandas()
logging.basicConfig(level=logging.INFO)


def load_data(dir_path: str) -> pd.DataFrame:
    '''
    Загружает предобработанные данные в виде csv из директории, 
    оставляет только непустые значения и шафлит, 
    чтобы подавать на обучение строки в случайном порядке.

    Args:
        dir_name (str): Путь до директории с csv файлами.


    Returns:
        pd.DataFrame: Датасет для обучения модели.
    '''

    df = pd.DataFrame()

    for file in tqdm(os.listdir(dir_path)):
        if file.endswith('.csv'):
            df_ = pd.read_csv(os.path.join(dir_path, file))
            df = pd.concat([df, df_])

    df = df.dropna().sample(frac=1)

    return df


def preprocessing(text: str) -> str:
    '''
    Чистит тексты сообщений от \n, лишних пробелов и отступов.

    TODO: сделать очистку от лишних символов, емоджи и т.д.
    Args:
        text (str): Текст для препроцессинга

    Returns:
        str: Очищенный текст.
    '''

    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)

    return text


def create_markup(row: pd.Series) -> str:
    '''
    Преобразует сообщения контекста и ответа в формат для обучения.

    Args:
        row (pd.Series): Строка датафрейма, содержащая context_3, context_2, context_1 и response столбцы.

    Returns:
        str: Строка, преобразованная в разметку диалога для обучения модели
    '''

    return '@@ПЕРВЫЙ@@ ' + row['context_3'] + \
           ' @@ВТОРОЙ ' + row['context_2'] + \
           ' @@ПЕРВЫЙ ' + row['context_1'] + \
           ' @@ПЕРВЫЙ ' + row['response']


def train_val_split(df: pd.DataFrame, val_size: float) -> tuple:
    '''
    Преобразует исходный датасет в нужный формат и делит его на обучающую и валидационную выборки.

    Args:
        df (pd.DataFrame): Датасет для разделения.
        val_size (float): Доля валидационной выборки относительно всего датасета.

    Returns:
        tuple: Кортеж из обучающего датасета и валидационного
    '''
    df_size = df.shape[0]
    train_number, val_number = round(df_size * (1 - val_size)), round(df_size * val_size)

    ds_train = Dataset.from_dict({'text': df['dialogue'].to_list()[:train_number]})
    ds_val = Dataset.from_dict({'text': df['dialogue'].to_list()[-val_number:]})

    return (ds_train, ds_val)


def encode(examples):
    '''
    Токенизирует данные для подачи модели.

    '''
    encoded = tokenizer(examples['text'],
                        truncation=True,
                        padding='max_length',
                        max_length=512)

    encoded['labels'] = encoded['input_ids'][:]

    return encoded


if __name__ == '__main__':
    df = load_data(cfg.data_dir_path)

    df['dialogue'] = df.progress_apply(lambda row: create_markup(row), axis=1)
    df['dialogue'] = df['dialogue'].progress_apply(lambda x: preprocessing(x))

    datasets = train_val_split(df, cfg.val_size)
    train_dataset, val_dataset = datasets[0], datasets[1]

    tokenizer = GPT2Tokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    model = GPT2LMHeadModel.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

    encoded_train = train_dataset.map(encode, batched=True)
    encoded_val = val_dataset.map(encode, batched=True)

    training_args = TrainingArguments(
        output_dir='ruDialoGPT_v2_medium',
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_dir=None,
        fp16=cfg.fp16,
        push_to_hub=True,
        hub_token=cfg.hf_token
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val
    )

    trainer.push_to_hub(cfg.model_source)
