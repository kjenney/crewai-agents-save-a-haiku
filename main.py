from crewai import Agent, Task, Process, Crew
from langchain_community.llms import Ollama

llm_mixtral = Ollama(model="dolphin-mistral")

writer = Agent(
    role="Haiku Expert",
    goal="Write Haiku",
    backstory="You love Haiku",
    verbose=True,
    allow_delegation=False,
    llm=llm_mixtral
)

write = Task(
    description="Write a short haiku about the ocean.",
    agent=writer,
    output_file='haiku.txt'
)

crew = Crew(
    agents=[writer],
    tasks=[write],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()

print("######################")
print(result)