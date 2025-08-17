import os
import time
from typing import Dict, List

from openai import OpenAI


class OpenAIClient:
    def __init__(self, model: str = 'gpt-4o') -> None:
        self._client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')).with_options(timeout=30)
        self.model = model

    def call(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        attempt = 0

        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=1.25
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    return str(e)

                # Exponential backoff: 4s, 16s, 64s ...
                sleep_s = 4 ** attempt
                time.sleep(sleep_s)
                continue
