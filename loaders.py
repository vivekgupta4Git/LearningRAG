from dotenv import load_dotenv
import os
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
#from langchain_core.prompts import format_document

model = ChatGoogleGenerativeAI(
    model = USER_AGENT,
    temperature = 0.7,
    top_p = 0.85
    
)

prompt = PromptTemplate(
    template="Write the breif summary of the website: {text}",
    input_variables=["text"],
)
#output_parser = StrOutputParser()


#loader = TextLoader("resume.txt", encoding="utf-8")
loader = WebBaseLoader("https://www.zyephr.com")
docs = loader.load()
#print(docs)
#print(docs[0].metadata)


output_parser = StrOutputParser()
chain = prompt | model | output_parser
print(chain.invoke({ "text": docs[0].page_content }))
