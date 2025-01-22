from langchain_openai import AzureChatOpenAI
import os
from crewai_tools import DirectoryReadTool, FileReadTool
from crewai_tools  import tool
from crewai import Agent, Task, Crew, LLM, Process
from textwrap import dedent
from typing_extensions import ClassVar
from crewai_tools import BaseTool
import json
import pymongo
from dotenv import load_dotenv
load_dotenv() 
llm = LLM(
        model="azure/Gpt4oSAtesting",
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        temperature=0
    )
data_retrieval_agent = Agent(
    role='Data Retrieval Specialist',
    goal='Be the best at getting all relevant info of {customer} from the database to help your team',
    backstory=dedent("""\
      You are ensuring to gather accurate and up-to-date data from the database
      which will help solving customers queries.
      """),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    memory=True
)

support_agent = Agent(
    role="Senior Support Representative",
	goal="Be the most friendly and helpful "
        "support representative in your team",
	backstory=dedent("""\
		You are working on providing support to {customer} which information provided by the data retrival specialist,
		a super important customer for your company.
		You need to make sure that you provide the best support!
		Make sure to provide full clear and complete answers,
    	and make no assumptions.
		"""),
	allow_delegation=False,
	verbose=True,
    llm=llm,
    memory=True
)

# allow delegation is true by default

support_quality_assurance_agent = Agent(
	role="Support Quality Assurance Specialist",
	goal="Get recognition for providing the "
    "best support quality assurance in your team",
	backstory=dedent("""\
		You are working with your team on a request from {customer} ensuring that the support representative is
		providing the best support possible.\n
		You need to make sure that the support representative is providing full clear and
		complete answers, and make no assumptions.
		"""),
	verbose=True,
    llm=llm,
    memory=True
)
directory_read_tool = DirectoryReadTool(directory='./content')
file_read_tool = FileReadTool()
myclient = pymongo.MongoClient(f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@cluster0.1vgus.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
mydb = myclient[os.getenv('DB_NAME')]
mycol = mydb["customer_detail"]

dic_data = {
  "customer_id": "CUST00124",
  "full_name": "Adam Ade",
  "email": "tommyade@example.com",
  "phone_number": "+1-555-123-4367",
  "address": "123 Maple Street, Springfield, IL 62704",
  "date_of_birth": "1995-07-24",
  "account_creation_date": "2023-01-15",
  "last_login_date": "2023-08-01",
  "subscription_plan": "Premium",
  "payment_method": "Credit Card",
  "preferred_language": "English",
  "support_history": "Last issue: Unable to change password, resolved on 2024-07-28",
  "loyalty_points": 1500,
  "special_notes": "Prefers email communication"
}
mycol.insert_one(dic_data)

class DatabaseRetrivalTool(BaseTool):
  name: str = "Database Retrival Tool"
  description: str = "Gets all the information of a customer from the database"
  myclient: ClassVar = pymongo.MongoClient(f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@cluster0.1vgus.mongodb.net/")
  mydb: ClassVar = myclient[os.getenv('DB_NAME')]

  def _run(self, text: str) -> str:
    docs: object = self.mydb["customer_detail"].find_one({'full_name': text})

    if docs is not None:
      del docs['_id'] # ObejectId is not serializable so remove it
      return json.dumps(docs)

    # failover incase name is not found
    return "No customer found"

retrival_tool = DatabaseRetrivalTool()

data_retrieval_task = Task(
   description=dedent("""\
      Gather all relevant {customer} data from the database, focusing
      on crucial data which will be great to know when addressing the
      customer's inquiry.
      """),
   expected_output=dedent("""\
      A comprehensive dataset of the customer's information.
      Highlighting key info of the customer that will be helpful
      to the team when addressing the customer's inquiry.
      """),
   tools=[retrival_tool],
   agent=data_retrieval_agent,
)

inquiry_resolution = Task(
   description=dedent("""\
      {customer} just reached out with a super important ask:\n
      {inquiry}\n\n
      {customer} is the one that reached out.
      Make sure to use everything you know
      to provide the best support possible.
      You must strive to provide a complete, clear
      and accurate response to the customer's inquiry.
      """),
   expected_output=dedent("""\
      A detailed, informative response to the
      customer's inquiry that addresses
      all aspects of their question.\n
      The response should include references
      to everything you used to find the answer,
      including external data or solutions.
      Ensure the answer is complete,
      leaving no questions unanswered, and maintain a helpful and friendly
      tone throughout.
      """),
   tools=[directory_read_tool, file_read_tool],
   agent=support_agent
)

quality_assurance_review = Task(
   description=dedent("""\
      Review the response drafted by the Senior Support Representative for {customer}'s inquiry.
      Ensure that the answer is comprehensive, accurate, and adheres to the
      high-quality standards expected for customer support.\n
      Verify that all parts of the customer's inquiry
      have been addressed thoroughly, with a helpful and friendly tone.\n
      Check for references and sources used to
      find the information,
      ensuring the response is well-supported and
      leaves no questions unanswered.
      """),
   expected_output=dedent("""\
      A final, detailed, and informative response
      ready to be sent to the customer.\n
      This response should fully address the
      customer's inquiry, incorporating all
      relevant feedback and improvements.\n
      Maintain a professional and friendly tone throughout.
      """),
   agent=support_quality_assurance_agent,
   output_file="response.md",
   # human_input=True
)

crew = Crew(
  agents=[data_retrieval_agent, support_agent, support_quality_assurance_agent],
  tasks=[data_retrieval_task, inquiry_resolution, quality_assurance_review],
  verbose=True,
  # process= Process.hierarchical,
)

inputs = {
    "customer": "Tommy Ade",
    "inquiry": "What is Crew AI?"
}
result = crew.kickoff(inputs=inputs)
