import re

def clean_return_message(message: str):
    if message.startswith("```json"):
        message = message[7:]
    if message.startswith("```"):
        message = message[3:]
    if message.endswith("```"):
        message = message[:-3]
    return message.strip()