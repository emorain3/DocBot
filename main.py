from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

load_dotenv()
chat_api_key = os.getenv("CHAT_API_KEY")

### DEFINE SCHEMA FOR THE EXPECTED OUTPUT ###
class ResearchOutput_SingleDate(BaseModel):
    date: str = Field(..., description="the date the medical data was processed")
    medical_data:  dict[str, str] = Field(..., description="The medical data that was recorded")
    summary: str = Field(..., description="A brief summary of the findings")
    sources: list[str] = Field(..., description="The sources used to compile the provided information")
    tools_used: list[str] = Field(..., description="List of tools used during research")

class ResearchOutput_OverTime(BaseModel):
    date_range: list[str] = Field(..., description="the date range that the medical data was collected")
    medical_data:  dict[str, int] = Field(..., description="The medical data that was recorded")
    summary: str = Field(..., description="A brief summary of the findings")
    sources: list[str] = Field(..., description="The sources used to compile the provided information")
    tools_used: list[str] = Field(..., description="List of tools used during research")

### IMPORT LLM ###
llm_openAI = ChatOpenAI(model_name="gpt-5-nano")
# llm_anthropic = ChatAnthropic(model="claude-haiku", temperature=0)

### CALL AND RESPONSE - example ###
# question = "What is the US State with the highest population?"
# response = llm_openAI.invoke(question)
# print(response)


### REFINED CALL AND RESPONSE ###
parser = PydanticOutputParser(pydantic_object=ResearchOutput_SingleDate)
# print("parser: ", parser)
# print("----------- format_instructions: -------------")
# print(parser.get_format_instructions())




prompt = ChatPromptTemplate.from_messages(
    [
    (
        "system", 
        """
        You are a medical research assistant. 
        Your task is to gather and analyze medical data based on user queries. 
        Use reliable sources and tools to compile accurate information. 
        Provide your findings in a structured format including date, medical data, summary, sources, and tools used.
        Wrap your response in the following format: \n{format_instructions}
        """
     ),
    ("placeholder", "{chat_history}"),("human", "{query}"), ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions = parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm_openAI,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "Analyze the trends in blood pressure readings over the past year for patients aged 40-60."})
print("* * * raw_response: ", raw_response)

try:
    structured_response = parser.parse(raw_response['output']['text'])
    print("* * * structured_response: ", structured_response)
except ValidationError as e:
    print("Validation error. :", e)   