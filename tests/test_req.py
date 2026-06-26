import os
import requests
from dotenv import load_dotenv

load_dotenv('.env')
api_key = os.environ.get("MISTRAL_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "ministral-8b-latest",
    "training_files": [{"file_id": "ac241613-67c3-409d-86a1-560b5e4a4883", "weight": 1}],
    "hyperparameters": {
        "training_steps": 100,
        "learning_rate": 0.0001
    },
    "job_type": "instruct"
}

res = requests.post("https://api.mistral.ai/v1/fine_tuning/jobs", headers=headers, json=payload)
print(res.status_code, res.text)
