import os
from typing import Any, Dict
import mosec
from .model import AbacusLLM

class AbacusWorker(mosec.Worker):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("ABACUS_API_KEY")
        project_id = os.getenv("ABACUS_PROJECT_ID")
        if not api_key or not project_id:
            raise ValueError("ABACUS_API_KEY and ABACUS_PROJECT_ID must be set")
        self.model = AbacusLLM(api_key, project_id)

class AbacusInference(AbacusWorker):
    def forward(self, request: Dict) -> Dict:
        try:
            if request.get("messages"):
                return self.model.chat(request["messages"])
            else:
                return self.model.completion(request["prompt"])
        except Exception as e:
            return {"error": str(e)}

class AbacusResponseFormatter(mosec.Worker):
    def forward(self, response: Dict) -> Dict:
        if "error" in response:
            return {"error": response["error"]}
        return response

def create_server():
    return mosec.Server(
        mosec.Chain([
            AbacusInference(),
            AbacusResponseFormatter()
        ])
    )
