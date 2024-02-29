from crewai import Agent, Task, Process, Crew
from langchain_community.llms import Ollama
from crewai_tools import FileReadTool

file_read_tool = FileReadTool()
llm_mixtral = Ollama(model="dolphin-mistral")

writer = Agent(
    role="Haiku Expert",
    goal="Write Haiku",
    backstory="You love Haiku",
    verbose=True,
    allow_delegation=False,
    llm=llm_mixtral
)

reviewer = Agent(
    role="Haiku Expert",
    goal="Review Haikus",
    backstory="You review Haikus",
    verbose=True,
    allow_delegation=False,
    llm=llm_mixtral
)

write = Task(
    description="Write a short haiku about the ocean.",
    agent=writer,
    output_file='haiku.txt'
)

review = Task(
    description="Read a haiku from haiku.txt and let me know if it's actually a haiku.",
    agent=reviewer,
    tools=[file_read_tool]
)

crew = Crew(
    agents=[writer,reviewer],
    tasks=[write,review],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()

print("######################")
print(result)