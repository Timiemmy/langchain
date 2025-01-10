from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch # Langchain built-in vector store
from IPython.display import display, Markdown
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator

file = 'your csv or other file'
loader = CSVLoader(file_path=file)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one." # or other question you want to ask.

llm_replacement_model = OpenAI(temperature=0,
                               model='gpt-3.5-turbo-instruct')

response = index.query(query,
                       llm=llm_replacement_model)
display(Markdown(response))


# Step by Step
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)

docs = loader.load()

print(docs[0])


from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")

print(len(embed))

print(embed[:5])

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"

docs = db.similarity_search(query)

print(len(docs))

print(docs[0])


retriever = db.as_retriever()
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0301")

qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

display(Markdown(response))

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)

display(Markdown(response))

response = index.query(query, llm=llm)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])


