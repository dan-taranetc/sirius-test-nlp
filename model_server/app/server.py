from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelWithLMHead
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained('/tmp/model')
model = AutoModelWithLMHead.from_pretrained('/tmp/model')


class UserRequest(BaseModel):
    query: str
    context: dict


def format_input(query: str, context: dict) -> str:
    context_1 = context.get('message_1')
    context_2 = context.get('message_2')
    context_3 = context.get('message_3')

    input_query = '@@ПЕРВЫЙ@@ '
    flag = True
    for message in [context_1, context_2, context_3, query]:
        if message:
            input_query += f'{message} {"@@ВТОРОЙ@@" if flag else "@@ПЕРВЫЙ@@"} '
            flag = not flag

    logging.warn(input_query)
    return input_query.strip()


@app.post("/answer")
def generate_answer(data: UserRequest):
    input_query = format_input(data.query, data.context)
    inputs = tokenizer(input_query, return_tensors='pt')
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=3,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        max_new_tokens=40
    )
    answer = tokenizer.decode(generated_token_ids[0]).split('@')
    return {"answer": answer[8]}
