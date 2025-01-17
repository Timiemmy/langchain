'''
Evaluation is the process of assessing the performance and effectiveness of your LLM-powered applications. 
It involves testing the model's responses against a set of predefined criteria or benchmarks
 to ensure it meets the desired quality standards and fulfills the intended purpose.
'''

# To evaluate, you need to have a chain or what you want to evaluate.
# We'll be evaluating the former document Q&A we created before.

from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation.qa import QAGenerateChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

# account for deprecation of LLM model
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

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])

llm = ChatOpenAI(temperature=0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"
    }
)

#Coming up with test datapoints
data[10]
data[11]

# Hard-coded examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
] # These are questions used in evaluating that file. You decide the questions you want to ask. But this is manual and not automated.

# Below is automated.
# LLM-Generated examples
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
# the warning below can be safely ignored
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

new_examples[0]
data[0]

# Combine examples
examples += new_examples
qa.run(examples[0]["query"])


# Manual Evaluation
import langchain
langchain.debug = True # setting this to True lets you see the things going on inside the evaluation process. 

qa.run(examples[0]["query"]) # This is the rerun of the qa above. 
#Sometimes it's not the llm that has the problem in returning right answer but the retrieval process.
# You'll get an overview of how many token you're using
# Turn off the debug mode
langchain.debug = False


# LLM assisted evaluation
predictions = qa.apply(examples) # This prints out the examples we have. You nay not have this much example. Itis just for tutorial.

llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)

graded_outputs = eval_chain.evaluate(examples, predictions)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()

graded_outputs[0]





