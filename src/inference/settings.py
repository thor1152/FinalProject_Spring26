from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    queue_url: str
    bucket_name: str = "seis765-finalproject-560852306721-us-east-1-an"
    model_key: str
    region_name: str = "us-east-1"
    poll_wait_time: int = 20
    max_messages: int = 10

settings = Settings()