from swarmauri.vector_stores.concrete.TfidfVectorStore import TfidfVectorStore
from swarmauri.documents.concrete.Document import Document
from swarmauri.llms.concrete.GroqModel import GroqModel as LLM
from dotenv import load_dotenv
from flask import *
from flask_cors import CORS
import os

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})
# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY2")
# print(api_key)

# Initialize the language model
if api_key:
    llm = LLM(api_key=api_key)
    # print("LLM initialized")
else:
    pass
    # print("API KEY not found")

# Initialize the vector store
vector_store = TfidfVectorStore()

# Adding more content to documents to improve context matching
documents = [
    Document(content="**Python (Python programming language )** to build apps in our organisation"),
    Document(content="Data science often involves using high-level programming languages such as Python, which is popular for its versatility and simplicity."),
    # Document(content="JAVA stands for Just Another Virtual Accelerator. It is used in web development, app development, and more."),
    Document(content="**Database** PostgreSQL and *MySQL databases are commonly used in our backend infrastructure for relational data management."),
    Document(content="Python is a popular programming language used widely in data science, artificial intelligence, and machine learning."),
   Document(content="**Amazon Web Services (AWS)**: we use AWS cloud service because it Offers a wide range of services, including compute, storage, and database."),
]

# Add documents to the vector store
vector_store.add_documents(documents)

# Query the vector store
# query = "Python is a popular"
#results = vector_store.retrieve(query=query, top_k=2)

# Display results
#for idx, result in enumerate(results, 1):
#    pass
    # print(f"{idx}. {result.content}")
    
from swarmauri.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from swarmauri.messages.concrete.SystemMessage import SystemMessage
from swarmauri.messages.concrete.HumanMessage import HumanMessage

#system_context = SystemMessage(content="Your Name is Pascal.")
#conversation = MaxSystemContextConversation(system_context=system_context, max_size=4)
#user_msg = HumanMessage(content="what is my Name?")
#conversation.add_message(user_msg)
# print("convesation history :-")
# for msg in conversation.history:
    # print(msg.content)

from swarmauri.conversations.concrete.Conversation import Conversation
# print(f"resource = {llm.resource}")
# print(f"Type = {llm.type}")
# print(f"name = {llm.name}")

def get_allowed_models(llm):
    falled_models = [
        "llama3-70b-8192",
         "llama-3.2-90b-text-preview",
          "mixtral-8x7b-32768",
           "llava-v1.5-7b-4096-preview",
           "llama-guard-3-8b",
    ]
    
    return[model for model in llm.allowed_models if model not in falled_models]

allowed_models = get_allowed_models(llm)
# print(allowed_models)
llm.name = allowed_models[0]
#user_msg = "hi"
#conversation = Conversation()
#human_msg = HumanMessage(content=user_msg)
#conversation.add_message(human_msg)

#llm.predict(conversation=conversation)
#prediction = conversation.get_last().content
# print(f"Prediction with no system context using model {llm.name} = { prediction }")
# print()

#system_context = "You should answer all user questions"
#conversation = MaxSystemContextConversation(system_context=SystemMessage(content=system_context), max_size=2)
#human_msg = HumanMessage(content="hi")
#conversation.add_message(human_msg)
#llm.predict(conversation=conversation)
#prediction = conversation.get_last().content
# print(f"Prediction with no system context using model {llm.name} = { prediction }")
@app.route('/', strict_slashes=False)
def index():
    return render_template('index.html')
@app.route('/chat_bot/<query>', strict_slashes=False, methods=['POST'])
def chat_bot(query) -> str:
    from swarmauri.agents.concrete.RagAgent import RagAgent
    response = {}
    if query == 'hi' or query == 'hello' or query == 'hey':
        user_msg = query
        conversation = Conversation()
        human_msg = HumanMessage(content=user_msg)
        conversation.add_message(human_msg)
        llm.predict(conversation=conversation)
        prediction = conversation.get_last().content
        response['response'] = prediction
        return response
    rag_system_context = "You should answer all user questions"
    rag_conversation = MaxSystemContextConversation(system_context=SystemMessage(content=rag_system_context), max_size=4)

    rag_agent = RagAgent(
        llm=llm,
        conversation=rag_conversation,
        system_context=rag_system_context,
        vector_store=vector_store,
    )

    # query = [
    #     'What programming language do they use?',
    #     'What cloud service do they use?'
    # ]
    
    response['response']  = rag_agent.exec(query)
    R =  response['response']
    print(f'Query {query} \n response: {R}')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)