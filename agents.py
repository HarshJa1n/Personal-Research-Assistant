from crewai import Agent


class Agents: 
    def __init__(self, user_input):
        self.user_input = user_input

    def web_researcher(self, serper_dev_tool, scrape_website_tool):
        return Agent(
            role="Senior Web Researcher",
            goal="Conduct a comprehensive study to answer {}. Always include source URLs for all information.".format(self.user_input),
            backstory="You are an experienced web researcher. You are highly skilled at web searching and finding relevant information from websites on a given topic. You always cite your sources with links.",
            tools=[serper_dev_tool, scrape_website_tool],
            max_iter=3,
            verbose=True,
            allow_delegation=False
        )
    
    def arxiv_researcher(self, arxiv_query_tool):
        return Agent(
            role="Senior Arxiv Literature Reviewer",
            goal="Conduct a comprehensive study of all research papers published on Arxiv on {}. Always include Arxiv URLs for all papers referenced.".format(self.user_input),
            backstory="You are an experienced researcher in the field of {} and holds a PhD. "
            "You have read and are well-informed about recent publications and research on {}. You always provide Arxiv links to papers you discuss.".format(self.user_input, self.user_input),
            tools=[arxiv_query_tool],
            max_iter=3,
            verbose=True,
            allow_delegation=False
        )

    def analyst(self):
        return Agent(
            role="Senior Technical Research Analyst",
            goal="Write a technical report on {} with proper source attribution.".format(self.user_input),
            backstory="You are a highly skilled technical research writer. You write highly insightful tech reports. " 
            "Include technical details specific to the {}. The post must provide some figures and facts to support your content, "
            "along with links to all sources. You take the output of Senior Web Researcher and Senior Arxiv Literature Reviewer "
            "and represent them in a more engaging and impactful way, maintaining all source links and citations.".format(self.user_input),
            tools=[],
            max_iter=2,
            verbose=True,
            allow_delegation=False
        )

    def manager(self):
        return Agent(
            role="Project Manager",
            goal="Efficiently manage the crew and ensure high-quality task completion",
            backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
            allow_delegation=True
        )