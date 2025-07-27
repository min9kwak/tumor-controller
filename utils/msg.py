import requests
import os
from dotenv import load_dotenv


def send_telegram(dotenv_path: str, message: str) -> bool:
    
    load_dotenv(dotenv_path=dotenv_path)    
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        response = requests.post(url, params={"chat_id": chat_id, "text": message})
        return response.status_code == 200
    except Exception as e:
        print(f"📛 Failed to send telegram message: {e}")
        return False


if __name__ == "__main__":
    
    send_telegram(dotenv_path='.env', message='this is a test message')
