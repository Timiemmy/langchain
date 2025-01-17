# account for deprecation of LLM model
from langchain.python import PythonREPL
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent
from langchain.agents.agent_toolkits import create_python_agent
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


#!pip install -U wikipedia
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model=llm_model)
tools = load_tools(["llm-math","wikipedia"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True)

    
agent("What is the 25% of 300?")
# Output
'''
Entering new AgentExecutor chain...
Thought: We can use the calculator tool to find 25% of 300.
Action:
```
{
  "action": "Calculator",
  "action_input": "25% of 300"
}
```
Observation: Answer: 75.0
Thought:Final Answer: 75.0

> Finished chain.
{'input': 'What is the 25% of 300?', 'output': '75.0'}
'''

# Wikipedia example

question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 

print(result)

#Python Agent
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)

customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]

agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")


# View detailed outputs of the chains
import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
langchain.debug=False


# Define your own tool
#!pip install DateTime

from langchain.agents import tool
from datetime import date


@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

'''
Note:
The agent will sometimes come to the wrong conclusion (agents are a work in progress!).
If it does, please try running it again.
'''

try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")


