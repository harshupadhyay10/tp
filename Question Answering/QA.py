from pydantic import BaseModel
from pydantic.env_settings import SecretsSettingsSource

class QA(BaseModel):
    context: str
    question: str