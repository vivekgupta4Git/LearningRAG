from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")

model = ChatGoogleGenerativeAI(
    model = USER_AGENT,
    temperature = 0.7,
    top_p = 0.85
)

result = model.invoke("Write the breif summary of the website: https://www.zyephr.com")
print(result.content)
#the output is not what we expected