import logging
import openai
from openai import AsyncOpenAI
import asyncio
import json
import os
import re
import emoji
import unicodedata
import random
import groq
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import (Message, CallbackQuery, BotCommand, MenuButtonCommands,
                           InputMediaPhoto, InputMediaVideo)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import State, StatesGroup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# =========================
# config.py должен содержать:
# TELEGRAM_TOKEN, OPENAI_API_KEY, ADMINS = [...]
# =========================
from config import TELEGRAM_TOKEN, GROQ_API_KEY, ADMINS

logging.basicConfig(level=logging.INFO)

# Инициализируем OpenAI
groq_client = groq.Groq(api_key=GROQ_API_KEY)


# Разрешённые эмодзи
emogies = ["❤️","👍","👎","🤣","😢","🔥","🤬","🙏","😱"]

# Инициализируем бота
bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher(storage=MemoryStorage())

CHANNELS_FILE = 'channels.json'

# =========================
# Чтение/запись каналов
# =========================
def save_channels():
    with open(CHANNELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(CHANNELS_DB, f, ensure_ascii=False, indent=2)

def load_channels():
    global CHANNELS_DB
    if os.path.exists(CHANNELS_FILE):
        with open(CHANNELS_FILE, 'r', encoding='utf-8') as f:
            CHANNELS_DB = json.load(f)
    else:
        CHANNELS_DB = []

# Загрузим каналы при старте
CHANNELS_DB = []
load_channels()

# =========================
# Утилиты для очистки текста
# =========================
def remove_telegram_links(text: str) -> str:
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text: str) -> str:
    return re.sub(r'#(\w+)', '', text)

def remove_urls(text: str) -> str:
    return re.sub(r'http\S+', '', text)

def strip_variation_selectors(s: str) -> str:
    return ''.join(
        ch for ch in s
        if not unicodedata.name(ch, "").startswith('VARIATION SELECTOR')
    )

def is_emoji(ch: str) -> bool:
    base_ch = strip_variation_selectors(ch)
    return base_ch in emoji.EMOJI_DATA

def handle_reaction_emojis(text: str) -> str:
    """
    Только последние 3 строки:
      - Заменяем «неразрешённые» эмодзи на разрешённые.
    """
    if not text:
        return text

    lines = text.split('\n')
    start_idx = max(0, len(lines) - 3)

    used_emojis = set()
    for i in range(start_idx, len(lines)):
        new_line = []
        for ch in lines[i]:
            if is_emoji(ch):
                base_ch = strip_variation_selectors(ch)
                if base_ch in emogies:
                    new_line.append(base_ch)
                else:
                    available = list(set(emogies) - used_emojis)
                    pick = random.choice(available) if available else random.choice(emogies)
                    new_line.append(pick)
                    used_emojis.add(pick)
            else:
                new_line.append(ch)
        lines[i] = ''.join(new_line)

    return '\n'.join(lines)

def has_reactions_in_last_3_lines(text: str, allowed_emojis=None) -> bool:
    if not text or not allowed_emojis:
        return False

    lines = text.split('\n')
    last_3 = lines[-3:] if len(lines) >= 3 else lines

    pattern = r'^(' + '|'.join(map(re.escape, allowed_emojis)) + r')\s*[-—]\s+.+'
    for line in last_3:
        if re.search(pattern, line.strip()):
            return True
    return False

# =========================
# Перефразирование
# =========================
async def paraphrase_text(text: str) -> str:
    """
    1) Удаляем @, хэштеги, ссылки
    2) Удаляем последнюю строку (демонстрация)
    3) handle_reaction_emojis
    4) Проверяем, есть ли реакции -> формируем system_prompt
    5) Use Groq for paraphrasing
    """
    text = text.strip()
    if not text:
        return ""

    lines = text.split('\n')
    if lines:
        text = '\n'.join(lines[:-1]).strip()

    # 1) Очистка
    text = remove_telegram_links(text)
    text = remove_hashtags(text)
    text = remove_urls(text)

    # 3) Заменяем недопустимые эмодзи
    text = handle_reaction_emojis(text)

    # 4) Проверяем, есть ли реакции
    has_reactions = has_reactions_in_last_3_lines(text, emogies)
    
    if has_reactions:
        system_prompt = (
            "Ты — помощник, который перефразирует текст на русском языке. "
            "Сохраняй все эмодзи и их порядок, а также структуру абзацев. "
            "Реакции уже есть, не добавляй новых."
        )
    else:
        system_prompt = (
        "Ты — помощник, который перефразирует текст на русском языке. "
        "Сохраняй все эмодзи и не меняй их порядок, а также структуру абзацев. "
        "\n\nЕсли в последних 3 строках нет эмодзи из списка: ❤️, 👍, 👎, 🤣, 😢, 🔥, 🤬, 🙏, 😱, "
        "добавь в самом конце ровно 3 новые строки реакций. "
        "Каждая строка должна начинаться с уникального эмодзи (из этого списка), "
        "после эмодзи идет дефис и короткий комментарий в скобках.\n\n"
        "Пример:\n"
        "❤️ - (одобрение)\n"
        "👎 - (отрицание)\n"
        "😱 - (удивление)\n\n"
        "Только три разных эмодзи, обязательно из списка.\n"
    )

    # 5) Запрос к Groq
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Ошибка при обращении к Groq: {e}")
        return text

# =========================
# FSM
# =========================
class AddChannelStates(StatesGroup):
    waiting_for_channel_username = State()
    waiting_for_custom_link = State()
    waiting_for_display_name = State()

class PostStates(StatesGroup):
    waiting_for_channel_selection = State()

# =========================
# Проверка на админа
# =========================
def admin_only(func):
    async def wrapper(message: Message, state: FSMContext):
        if message.from_user.id not in ADMINS:
            await message.answer("Вы не админ!")
            return
        return await func(message, state)
    return wrapper

# =========================
# Команды
# =========================

@dp.message(Command("start"))
@admin_only
async def cmd_start(message: Message, state: FSMContext):
    """
    Команда /start
    """
    await state.clear()
    text = (
        "<b>Доступные команды для управления ботом:</b>\n\n"
        "/addchannel – Добавить канал (через @username)\n"
        "/listchannels – Показать все каналы\n"
        "/delchannel – Удалить канал\n\n"
        "Отправьте альбом (или одиночное сообщение) с фото/видео/текстом, "
        "и бот предложит выбрать каналы."
    )
    await message.answer(text)

@dp.message(Command("listchannels"))
@admin_only
async def cmd_listchannels(message: Message, state: FSMContext):
    """
    Команда /listchannels
    """
    if not CHANNELS_DB:
        await message.answer("<b>Список каналов пуст.</b>")
        return

    lines = []
    for ch in CHANNELS_DB:
        user = ch["username"]
        title = ch["title"]
        display_name = ch.get("display_name","")
        lines.append(f"• {title} ({user}) — Display: {display_name}")

    text = "<b>Список каналов:</b>\n\n" + "\n".join(lines)
    await message.answer(text)

@dp.message(Command("addchannel"))
@admin_only
async def cmd_addchannel(message: Message, state: FSMContext):
    """
    Команда /addchannel
    """
    await state.set_state(AddChannelStates.waiting_for_channel_username)
    await message.answer(
        "<b>Добавление нового канала:</b>\n\n"
        "Введите @username канала (например, @MyPublicChannel)."
    )

@dp.message(AddChannelStates.waiting_for_channel_username, F.text)
async def process_channel_username(message: Message, state: FSMContext):
    """
    Обрабатываем ввод @username канала
    """
    username = message.text.strip()
    if not username.startswith('@'):
        await message.answer("<b>Имя канала должно начинаться с @!</b>")
        return

    try:
        chat = await bot.get_chat(username)
        channel_title = chat.title or username

        # Проверка на дубликаты
        for c in CHANNELS_DB:
            if c["username"].lower() == username.lower():
                await message.answer("<b>Этот канал уже есть!</b>")
                await state.clear()
                return

        await state.update_data(channel_username=username, channel_title=channel_title)
        await state.set_state(AddChannelStates.waiting_for_custom_link)
        await message.answer(
            f"<b>Канал {channel_title} ({username}) найден!</b>\n\n"
            "Теперь введите ссылку, которую будем добавлять в конце постов (пример: https://example.com)."
        )
    except Exception as e:
        await message.answer(f"<b>Ошибка при добавлении канала:</b> {e}")

@dp.message(AddChannelStates.waiting_for_custom_link, F.text)
async def process_custom_link(message: Message, state: FSMContext):
    """
    Вводим ссылку для канала
    """
    await state.update_data(custom_link=message.text.strip())
    await state.set_state(AddChannelStates.waiting_for_display_name)
    await message.answer("<b>Теперь введите полное название канала (display_name) для конца постов:</b>")

@dp.message(AddChannelStates.waiting_for_display_name, F.text)
async def process_display_name(message: Message, state: FSMContext):
    """
    Вводим display_name
    """
    display_name = message.text.strip()
    data = await state.get_data()
    username = data["channel_username"]
    channel_title = data["channel_title"]
    custom_link = data["custom_link"]

    CHANNELS_DB.append({
        "username": username,
        "title": channel_title,
        "custom_link": custom_link,
        "display_name": display_name
    })
    save_channels()

    await state.clear()
    await message.answer(
        f"<b>Канал {channel_title} ({username}) добавлен!</b>\n"
        f"Ссылка: {custom_link}, DisplayName: {display_name}"
    )

@dp.message(Command("delchannel"))
@admin_only
async def cmd_delchannel(message: Message, state: FSMContext):
    """
    Команда /delchannel
    """
    if not CHANNELS_DB:
        await message.answer("<b>Список каналов пуст.</b>")
        return

    builder = InlineKeyboardBuilder()
    for ch in CHANNELS_DB:
        builder.button(
            text=f"Удалить {ch['title']} ({ch['username']})",
            callback_data=f"delete_channel:{ch['username']}"
        )
    builder.adjust(1)
    await message.answer("<b>Выберите канал для удаления:</b>", reply_markup=builder.as_markup())

@dp.callback_query(F.data.startswith("delete_channel:"))
@admin_only
async def callback_delete_channel(callback: CallbackQuery, state: FSMContext):
    """
    Обработка удаления канала
    """
    username = callback.data.split(":", maxsplit=1)[1]

    for i,ch in enumerate(CHANNELS_DB):
        if ch['username'] == username:
            del CHANNELS_DB[i]
            save_channels()
            await callback.answer(f"Канал {username} удалён.")
            await callback.message.edit_text("<b>Канал успешно удалён!</b>")
            return

    await callback.answer("<b>Канал не найден!</b>", show_alert=True)


# =========================
# ЛОГИКА АЛЬБОМОВ (media_group_id) - вручную
# =========================

media_groups_buffer = {}  # {media_group_id: {...}}
ALBUM_TIMEOUT = 2.0  # сек.

async def finish_album(media_group_id: str, last_message: Message, state: FSMContext):
    """
    Вызывается по таймеру, когда решили, что альбом собран
    """
    await asyncio.sleep(ALBUM_TIMEOUT)
    data = media_groups_buffer.pop(media_group_id, None)
    if not data:
        return

    photos = data["photos"]
    videos = data["videos"]
    captions = data["captions"]

    original_text = "\n".join(captions).strip()
    paraphrased_text = ""
    if original_text:
        paraphrased_text = await paraphrase_text(original_text)

    # Сохраняем в FSM
    await state.update_data(
        paraphrased_text=paraphrased_text,
        photos=photos,
        videos=videos
    )

    total_files = len(photos)+len(videos)
    preview_text = "<b>Предпросмотр альбома:</b>\n\n"
    preview_text += f"• Элементов: {total_files}\n"
    preview_text += f"\n{paraphrased_text or '(пусто)'}"

    await state.set_state(PostStates.waiting_for_channel_selection)

    builder = InlineKeyboardBuilder()
    for ch in CHANNELS_DB:
        builder.button(
            text=f"{ch['title']}",
            callback_data=f"select_channel:{ch['username']}"
        )
    builder.button(text="✅ Отправить", callback_data="send_post")
    builder.adjust(1)

    await last_message.answer(
        text=preview_text + "\n\n<b>Выберите каналы для отправки:</b>",
        reply_markup=builder.as_markup()
    )

@dp.message()
@admin_only
async def handle_message(message: Message, state: FSMContext):
    """
    Обработка ВСЕХ сообщений:
    - Если есть media_group_id => Альбом
    - Иначе => Одиночное сообщение
    """
    mg_id = message.media_group_id
    if mg_id:
        # альбом
        if mg_id not in media_groups_buffer:
            media_groups_buffer[mg_id] = {
                "photos": [],
                "videos": [],
                "captions": [],
                "timer": None
            }
        data = media_groups_buffer[mg_id]

        # Собираем фото/видео
        if message.photo:
            data["photos"].append(message.photo[-1].file_id)
        if message.video:
            data["videos"].append(message.video.file_id)
        if message.caption and message.caption.strip():
            data["captions"].append(message.caption)

        # Перезапускаем таймер
        if data["timer"]:
            data["timer"].cancel()
        data["timer"] = asyncio.create_task(finish_album(mg_id, message, state))

    else:
        # одиночное сообщение
        photos = []
        videos = []
        if message.photo:
            photos.append(message.photo[-1].file_id)
        if message.video:
            videos.append(message.video.file_id)

        original_text = message.caption or message.text or ""
        paraphrased_text = ""
        if original_text.strip():
            paraphrased_text = await paraphrase_text(original_text)

        await state.update_data(
            paraphrased_text=paraphrased_text,
            photos=photos,
            videos=videos
        )

        total_files = len(photos)+len(videos)
        preview_text = "<b>Предпросмотр:</b>\n\n"
        if total_files>1:
            preview_text += f"• Элементов: {total_files}\n"
        elif len(photos)==1:
            preview_text += "• Фото\n"
        elif len(videos)==1:
            preview_text += "• Видео\n"
        else:
            preview_text += "• Без медиа\n"

        preview_text += f"\n{paraphrased_text or '(пусто)'}"

        await state.set_state(PostStates.waiting_for_channel_selection)
        builder = InlineKeyboardBuilder()
        for ch in CHANNELS_DB:
            builder.button(
                text=f"{ch['title']}",
                callback_data=f"select_channel:{ch['username']}"
            )
        builder.button(text="✅ Отправить", callback_data="send_post")
        builder.adjust(1)

        await message.answer(
            text=preview_text + "\n\n<b>Выберите каналы для отправки:</b>",
            reply_markup=builder.as_markup()
        )

# =========================
# Выбор каналов и отправка
# =========================
@dp.callback_query(PostStates.waiting_for_channel_selection, F.data.startswith("select_channel:"))
async def cb_select_channel(callback: CallbackQuery, state: FSMContext):
    username = callback.data.split(':', maxsplit=1)[1]
    data = await state.get_data()
    selected_channels = data.get("selected_channels", [])

    if username in selected_channels:
        selected_channels.remove(username)
        await callback.answer(f"Канал {username} убран.")
    else:
        selected_channels.append(username)
        await callback.answer(f"Канал {username} добавлен.")

    await state.update_data(selected_channels=selected_channels)


@dp.callback_query(PostStates.waiting_for_channel_selection, F.data=="send_post")
async def cb_send_post(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    paraphrased_text = data.get("paraphrased_text","")
    photos = data.get("photos",[])
    videos = data.get("videos",[])
    selected_channels = data.get("selected_channels",[])

    if not selected_channels:
        await callback.answer("Вы не выбрали ни одного канала!", show_alert=True)
        return

    all_files = []
    for p in photos:
        all_files.append(("photo", p))
    for v in videos:
        all_files.append(("video", v))

    for username in selected_channels:
        ch_data = next((c for c in CHANNELS_DB if c['username']==username), None)
        if not ch_data:
            logging.error(f"Channel {username} not found!")
            continue

        display_name = ch_data.get("display_name","")
        custom_link = ch_data.get("custom_link","")

        paragraphs = paraphrased_text.split('\n\n',1)
        if len(paragraphs)>1:
            first_p, rest = paragraphs
            post_text = f"<b>{first_p}</b>\n\n{rest}\n\n<a href=\"{custom_link}\">{display_name}</a>"
        else:
            post_text = f"<b>{paraphrased_text}</b>\n\n<a href=\"{custom_link}\">{display_name}</a>"

        try:
            if not all_files:
                # нет медиа
                await bot.send_message(username, post_text)
            elif len(all_files)==1:
                # один элемент
                t, f_id = all_files[0]
                if t=="photo":
                    await bot.send_photo(username, f_id, caption=post_text)
                else:
                    await bot.send_video(username, f_id, caption=post_text)
            else:
                # альбом
                media_group = []
                for i,(t,f_id) in enumerate(all_files):
                    if i==0:
                        if t=="photo":
                            media_group.append(InputMediaPhoto(media=f_id, caption=post_text))
                        else:
                            media_group.append(InputMediaVideo(media=f_id, caption=post_text))
                    else:
                        if t=="photo":
                            media_group.append(InputMediaPhoto(media=f_id))
                        else:
                            media_group.append(InputMediaVideo(media=f_id))

                await bot.send_media_group(username, media_group)
            await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Не удалось отправить в {username}: {e}")

    await state.clear()
    await callback.message.edit_text("<b>Пост успешно отправлен!</b>")
    await callback.answer()

# =========================
# Меню команд
# =========================
async def setup_menu_buttons():
    commands = [
        BotCommand(command="start", description="Начало"),
        BotCommand(command="addchannel", description="Добавить канал"),
        BotCommand(command="listchannels", description="Список каналов"),
        BotCommand(command="delchannel", description="Удалить канал"),
    ]
    await bot.set_my_commands(commands)
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands(type="commands"))

# =========================
# Точка входа
# =========================
async def main():
    try:
        await setup_menu_buttons()
        await dp.start_polling(bot)
    finally:
        save_channels()
        await bot.session.close()

if __name__=="__main__":
    asyncio.run(main())
