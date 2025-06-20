from dotenv import load_dotenv
import os

load_dotenv("entity extraction/.env")
print(os.getenv("DEEPSEEK_API_KEY"))