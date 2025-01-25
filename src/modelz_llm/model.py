import abacusai
from typing import Dict, List, Optional

class AbacusLLM:
    def __init__(self, api_key: str, project_id: str):
        self.client = abacusai.ApiClient(api_key)
        self.project_id = project_id

    def chat(self, messages: List[Dict], **kwargs):
        try:
            response = self.client.chat_completion(
                project_id=self.project_id,
                messages=messages,
                **kwargs
            )
            return {
                'choices': [{
                    'message': {
                        'role': 'assistant',
                        'content': response.choices[0].message.content
                    }
                }]
            }
        except Exception as e:
            raise RuntimeError(f"Abacus AI API error: {str(e)}")

    def completion(self, prompt: str, **kwargs):
        try:
            response = self.client.text_completion(
                project_id=self.project_id,
                prompt=prompt,
                **kwargs
            )
            return {
                'choices': [{
                    'text': response.choices[0].text
                }]
            }
        except Exception as e:
            raise RuntimeError(f"Abacus AI API error: {str(e)}")
