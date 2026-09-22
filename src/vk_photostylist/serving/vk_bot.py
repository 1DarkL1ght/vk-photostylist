import os

# import requests
import vk_api
from dotenv import load_dotenv
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
from vk_api.upload import VkUpload

load_dotenv()

VK_TOKEN = os.getenv("VK_TOKEN")
GROUP_ID = os.getenv("GROUP_ID")

# event
# <<class 'vk_api.bot_longpoll.VkBotMessageEvent'>
# ({'group_id': 241659236, 'type': 'message_new',
# 'event_id': '5dc45851882a0354db03c8bc5740477f3719b69e', 'v': '5.199',
# 'object': {'client_info': {'button_actions':
# ['text', 'vkpay', 'open_app', 'location', 'open_link',
# 'open_photo', 'callback', 'intent_subscribe', 'intent_unsubscribe'],
# 'keyboard': True, 'inline_keyboard': True,
# 'carousel': True, 'lang_id': 0},
# 'message': {'date': 1790052840, 'from_id': 677866122,
# 'id': 6, 'version': 10000009, 'out': 0,
# 'fwd_messages': [], 'important': False,
# 'is_hidden': False, 'attachments': [],
# 'conversation_message_id': 6, 'text': 'fffff',
# 'peer_id': 677866122, 'random_id': 0}}})>


def get_max_size_url_photo(photo_data: dict) -> str:
    sizes = photo_data.get("sizes", [])
    if not sizes:
        return ""
    max_size = max(sizes, key=lambda s: s.get("width", 0) * s.get("height", 0))
    return max_size.get("url", "")


def run_bot(vk_token: str, group_id: str):
    vk_session = vk_api.VkApi(token=vk_token)
    longpoll = VkBotLongPoll(vk_session, group_id)

    for event in longpoll.listen():
        if event.type == VkBotEventType.MESSAGE_NEW:
            message = event.object.message
            # text = event.message.text
            from_id = event.message.from_id

            attachments = message.get("attachments", [])

            photo_urls = []
            for attachment in attachments:
                if attachment.get("type") == "photo":
                    url = get_max_size_url_photo(attachment.get("photo"))
                    if url:
                        photo_urls.append(url)

            if photo_urls:
                response_to_user("Получен", from_id, vk_session, photo_urls[0])
            else:
                response_to_user("Пришлите фото", from_id, vk_session, None)


def response_to_user(
    message: str, from_id: str, vk_session: vk_api.VkApi, photo_path: str
):
    vk = vk_session.get_api()

    upload = VkUpload(vk_session)
    photo = upload.photo_messages(photos=photo_path)[0]
    attachment = f"photo{photo['owner_id']}_{photo['id']}"

    response = f"{message}"
    vk.messages.send(
        user_id=from_id, message=response, attachment=attachment, random_id=0
    )


run_bot(VK_TOKEN, GROUP_ID)
