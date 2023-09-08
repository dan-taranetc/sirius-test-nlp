from speech_decoding import decoder
from db_operations import get_context, update_context, clear_context
from aiogram import Bot, Dispatcher, executor, types
import os
import uuid
import requests
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

bot = Bot(token=os.getenv('BOT_TOKEN'))
dp = Dispatcher(bot)


# Стартовое сообщение бота
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    msg = '''
    Привет! Я - твой личный бот-собеседник DialoGPTv2. 
    Я был обучен поддерживать диалог с учетом контекста на многих программистских чатах в телеграмме и не только. 
    Просто начни разговор, и мы можем поговорить о чем угодно: от ответов на вопросы до обыденной беседы. 
    Чем могу помочь сегодня?
    
    P.S. Я также умею отвечать и на голосовые сообщения.
    '''
    await message.answer(msg)


# Help сообщение бота
@dp.message_handler(commands=['help'])
async def help_message(message: types.Message):
    msg = '''
    Для того чтобы воспользоваться мной просто отправь текстовое сообщение в чат и я отвечу на него.
    Также ты можешь записать голосовое сообщение, которое я расшифрую и также отвечу на него.
    Если ты хочешь очитить контекст, просто воспользуйся соответсвующей командой: `/clear_state`.
    '''
    await message.answer(msg)


# Чистка контекста пользователя
@dp.message_handler(commands=['clear_context'])
async def clear_user_context(message: types.Message):
    user_id = message['from']['id']
    clear_context(user_id)

    await message.answer('Контекст очищен.')


# Ответ бота и модели на текстовый запрос пользователя
@dp.message_handler(content_types=[types.ContentType.TEXT])
async def answer(message: types.Message):
    user_id = message['from']['id']
    context = get_context(user_id)

    response = requests.post('http://dialogpt_server:80/answer', json={'query': message.text, 'context': context})
    model_answer = response.json()['answer']

    await message.answer(str(response.json()['answer']))

    update_context(user_id, message.text, model_answer)


# Ответ бота и модели на голосовой запрос пользователя
@dp.message_handler(content_types=[types.ContentType.VOICE])
async def voice_answer(message: types.Message):
    try:
        filename = str(uuid.uuid4())

        voice = await message.voice.get_file()
        await bot.download_file(file_path=voice.file_path, destination=f"{filename}.ogg")

        text = decoder.voice_to_text(filename)
    except Exception:
        await message.answer('Не удалось распознать голосовое сообщение.')
    else:
        await message.answer(f'Распознанное сообщение: {text}')

        user_id = message['from']['id']
        context = get_context(user_id)

        response = requests.post('http://dialogpt_server:80/answer', json={'query': text, 'context': context})
        model_answer = response.json()['answer']

        await message.answer(str(response.json()['answer']))

        update_context(user_id, message.text, model_answer)


if __name__ == '__main__':
    logging.info('Starting bot')
    executor.start_polling(dp, skip_updates=True)
