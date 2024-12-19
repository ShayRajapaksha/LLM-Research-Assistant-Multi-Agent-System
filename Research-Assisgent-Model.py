from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json  


load_dotenv()

# API keys
open_api = os.getenv("OPENAPI_API_KEY")
serper_api = os.getenv("SERPER_API_KEY")

# LLM 
llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=open_api, temperature=0.7)

# Tools
search = SerperDevTool(api_key=serper_api)

# Researcher Agent
class ResearcherAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = self.load_memory()  # Load memory 

    def store_memory(self, key, value):
        self.memory[key] = value
        self.save_memory()  # Save 

    def retrieve_memory(self, key):
        return self.memory.get(key, None)

    def load_memory(self):
        try:
            with open(r"C:\Users\shayx\Desktop\ACA\Multi agent\Pyton working\researcher_memory.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def save_memory(self):
        with open(r"C:\Users\shayx\Desktop\ACA\Multi agent\Pyton working\researcher_memory.json", "w") as file:
            json.dump(self.memory, file)

    def conduct_research(self):
        # Consider past research
        previous_analysis = self.retrieve_memory("data_analysis")
        research_output = "Promising trends in quantum computing research..."
        if previous_analysis:
            research_output += f" Previous insights: {previous_analysis}"
        self.store_memory("last_research", research_output)
        return research_output


#Data Analysis Agent
class DataAnalysisAgent(Agent):
    def analyze_data(self, research_data):
        # Simulate data analysis
        analysis_output = "Key insights extracted from the research..."
        return analysis_output


#Summary Agent
class SummaryAgent(Agent):
    def summarize(self, research_output):
        # Simulate summarizing the research output
        summary_output = {
            "Main Topics": ["Quantum Computing Basics", "Latest Innovations"],
            "Subtopics": ["Quantum Algorithms", "Quantum Hardware"]
        }
        return summary_output


#Writer Agent
class WriterAgent(Agent):
    def write_speech(self, summary):
        # Simulate speech writing
        speech_output = "Engaging keynote speech based on the provided summary..."
        return speech_output


#Fact-Checking Agent
class FactCheckingAgent(Agent):
    def check_facts(self, speech):
        # Simulate fact-checking process
        verification_output = "All facts are verified; no corrections needed."
        return verification_output


# Initialize agents
user_goal = input("What is your goal for the research? ")

researcher = ResearcherAgent(
    llm=llm,
    role="Senior AI Researcher",
    goal=user_goal,
    backstory="You are a veteran  researcher that can research any topic on the internet.",
    allow_delegation=False,
    tools=[search],
    verbose=1,
)

data_analysis = DataAnalysisAgent(
    llm=llm,
    role="Data Analyst",
    goal="Analyze research findings for accuracy and trends.",
    backstory="You have a strong background in data science and statistical analysis.",
    allow_delegation=False,
    verbose=1,
)

summary_agent = SummaryAgent(
    llm=llm,
    role="Summary Specialist",
    goal="Summarize research findings for topics and main points ",
    backstory="You excel at distilling complex topics into digestible summaries.",
    allow_delegation=False,
    verbose=1,
)

writer = WriterAgent(
    llm=llm,
    role="Senior Speech Writer",
    goal="Write a knowledgeable note from the research data and summary.",
    backstory="You are a veteran research note writer with a background in clear, accessible writing.",
    allow_delegation=False,
    verbose=1,
)

fact_checker = FactCheckingAgent(
    llm=llm,
    role="Fact Checker",
    goal="Verify the accuracy of research notes.",
    backstory="You ensure the credibility of information presented.",
    allow_delegation=False,
    verbose=1,
)

# Create tasks for the agents
task1 = Task(
    description="Search the internet and find data on given user goal.",
    expected_output="A detailed explanation of each researched topic.",
    output_file="Researched-Data.txt",
    agent=researcher,
)

task2 = Task(
    description="Analyze the findings gathered by the Researcher Agent.",
    expected_output="Insights on the quality of the research results based on data analysis.",
    agent=data_analysis,
)

task3 = Task(
    description="Summarize the output from the Researcher Agent into main topics and subtopics.",
    expected_output="Main topics and subtopics from the research.",
    output_file="study-points.txt",
    agent=summary_agent,
)

task4 = Task(
    description="Write an engaging research note based on the research findings and summary topics.",
    expected_output="A detailed research note with an intro, body, conclusion, and important facts.",
    output_file="NOTE.txt",
    agent=writer,
)

task5 = Task(
    description="Verify the accuracy of the research note.",
    expected_output="Verification results and any necessary corrections.",
    agent=fact_checker,
)

task6 = Task(
    description="Check one or two small misleading facts from the research note",
    expected_output="One or two misleading facts from the research note.",
    agent=fact_checker,
)

task7 = Task(
    description="Revise the research note by removing the identified incorrect points provided by the fact checker agent .",
    expected_output="A corrected version of the research note.",
    output_file="FINAL_NOTE.txt",
    agent=writer,
)

#Create the crew 
crew = Crew(
    agents=[researcher, data_analysis, summary_agent, writer, fact_checker], 
    tasks=[task1, task2, task3, task4, task5, task6, task7],
    verbose=1,
)

# Start the crew and perform 
print("Conducting research...")
analysis_result = data_analysis.analyze_data("Research data here")
researcher.store_memory("data_analysis", analysis_result)  # Save 

new_research = researcher.conduct_research()  #Researcher uses saved memory
print("New Research Output:", new_research)

print("\nStarting the crew...")
print(crew.kickoff())
