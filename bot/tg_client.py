import json
import urllib.request
import os
from dotenv import load_dotenv

load_dotenv()


def make_request(method: str, **param) -> dict:
    json_data = json.dumps(param).encode('utf-8')
    request = urllib.request.Request(
        method='POST',
        url=f'{os.getenv("TG_URL")}/{method}',
        data=json_data,
        headers={
            "Content-Type": "application/json"
        }
    )

    with urllib.request.urlopen(request) as response:
        response_body = response.read().decode('utf-8')
        response_json = json.loads(response_body)
        assert response_json["ok"] is True
        return response_json["result"]


def getUpdates(offset: int) -> dict:
    return make_request('getUpdates', offset=offset)


def sendMessage(chat_id: int, text: str) -> dict:
    return make_request('sendMessage', chat_id=chat_id, text=text)


def getMe() -> dict:
    return make_request('getMe')


def getFile(file_id: str) -> dict:
    return make_request('getFile', file_id=file_id)


def download_file(file_path: str) -> bytes:
    """Скачивает файл с серверов Telegram"""
    url = f"https://api.telegram.org/file/bot{os.getenv('TG_BOT_TOKEN')}/{file_path}"
    with urllib.request.urlopen(url) as response:
        return response.read()