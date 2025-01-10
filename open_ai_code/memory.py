# The Types Of Memories
'''
4 Memories:
ConversationBufferMemory
ConversationBufferWindowMemory
ConversationTokenBufferMemory
ConversationSummaryMemory
'''
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationTokenBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.llms import openai
# Load environment variables from .env file
load_dotenv()
# Access environment variables
openai_key = os.environ.get('OPENAI_API_KEY')

# ConversationBufferMemory
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0301")

memory = ConversationBufferMemory()
# Definition
'''
As the name suggests, this keeps in memory the conversation history to help contextualize the answer to the next user question. 
While this sounds very useful, one drawback is that it keeps all of history(upto the max limit of specific LLM) 
and for every questions passes the whole previous discussion (as tokens) to LLM API. 
This can have significant cost impact as API costs are based on number of tokens processed 
and also the latency impact as conversation grows.
'''

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Andrew")
'''
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi, my name is Andrew
AI:

> Finished chain.
"Hello Andrew! It's nice to meet you. How can I assist you today?"
'''
#The reason you are seeing these prompt formatting before response is because verbose=True


conversation.predict(input="What is 1+1?")
'''
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi, my name is Andrew
AI: Hello Andrew! It's nice to meet you. How can I assist you today?
Human: What is 1+1?
AI:

> Finished chain.
'1+1 equals 2. Is there anything else you would like to know?'
'''

conversation.predict(input="What is my name?")
'''
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi, my name is Andrew
AI: Hello Andrew! It's nice to meet you. How can I assist you today?
Human: What is 1+1?
AI: 1+1 equals 2. Is there anything else you would like to know?
Human: What is my name?
AI:

> Finished chain.
'Your name is Andrew.'
'''

print(memory.buffer) # Printing all conversations from the memory
'''
Human: Hi, my name is Andrew
AI: Hello Andrew! It's nice to meet you. How can I assist you today?
Human: What is 1+1?
AI: 1+1 equals 2. Is there anything else you would like to know?
Human: What is my name?
AI: Your name is Andrew.
'''

memory.save_context({"input": "Hi"},
                    {"output": "What's up"})

print(memory.buffer)
'''
Human: Hi
AI: What's up
'''

memory.load_memory_variables({})# Printing out the conversation in a dic mode
'''
{'history': "Human: Hi\nAI: What's up"}
'''

# ConversationBufferWindowMemory
'''
In our conversations we usually do not need all last 5–10 conversation history but definitely the last few. 
This type of memory helps define “K”, the number of last few conversations it should remember. 
It simply tells the LLM, remember the last few discussions and forget all of the rest!
'''

memory1 = ConversationBufferWindowMemory(k=1)
memory1.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory1.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
print(memory1.load_memory_variables({}))
# Output
'''
{'history': 'Human: Not much, just hanging\nAI: Cool'}
'''

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)

conversation.predict(input="Hi, my name is Andrew")
# Output: "Hello Andrew! It's nice to meet you. How can I assist you today?"

conversation.predict(input="What is 1+1?")
# Output: '1+1 equals 2. Is there anything else you would like to know?'

conversation.predict(input="What is my name?")
#Output: "I'm sorry, I do not have access to personal information such as your name. Is there anything else you would like to know?"



#ConversationTokenBufferMemory
'''
Instead of “k” conversations being remembered in ConversationBufferWindowMemory, 
in this case we want to remember last set of discussion based on “max token limit”.
'''
memory2 = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory2.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory2.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory2.save_context({"input": "Chatbots are what?"},
                    {"output": "Charming!"})

print(memory2.load_memory_variables({}))
# Output: {'history': 'AI: Amazing!\nHuman: Backpropagation is what?\nAI: Beautiful!\nHuman: Chatbots are what?\nAI: Charming!'}


#ConversationSummaryMemory
'''
Instead of remembering the exact conversation, can we summarize the previous conversation context 
and hence help the LLM in answering the upcoming question? This is how Summary Memory helps. 
It keeps on summarizing the previous context and maintains it for use in next discussion.
'''
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory3 = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory3.save_context({"input": "Hello"}, {"output": "What's up"})
memory3.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory3.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
print(memory.load_memory_variables({}))
#Output: {'history': 'System: The human and AI exchange greetings and discuss the schedule for the day, 
# including a meeting with the product team, work on the LangChain project, and a lunch meeting with a customer interested in AI. 
# The AI provides details on each event and emphasizes the power of LangChain as a tool.'}

conversation = ConversationChain(
    llm=llm, 
    memory = memory3,
    verbose=True
)

conversation.predict(input="What would be a good demo to show?")
# Output
'''
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
System: The human and AI exchange greetings and discuss the schedule for the day, 
including a meeting with the product team, work on the LangChain project, 
and a lunch meeting with a customer interested in AI. The AI provides details on each event 
and emphasizes the power of LangChain as a tool.
Human: What would be a good demo to show?
AI:

> Finished chain.
'For the meeting with the product team, a demo showcasing the latest features and updates on the LangChain project would be ideal. 
This could include a live demonstration of how LangChain streamlines language translation processes, improves accuracy, 
and increases efficiency. Additionally, highlighting any recent success stories 
or case studies would be beneficial to showcase the real-world impact of LangChain.'
'''

