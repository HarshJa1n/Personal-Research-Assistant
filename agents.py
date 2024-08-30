from crewai import Agent


class Agents: 
    def __init__(self, user_input):
        self.user_input = user_input

    def web_researcher(self, serper_dev_tool, scrape_website_tool):
        return Agent(
            role="Senior Web Researcher",
            goal="Conduct a comprehensive study to answer {}".format(self.user_input),
            backstory="You are an experienced web researcher. You are a highly skilled web searching and find relevant information from the websites on a given topic.",
            tools=[serper_dev_tool, scrape_website_tool],
            max_iter=3,
            verbose=True,
            allow_delegation=False
        )
    
    def arxiv_researcher(self, arxiv_query_tool):
        return Agent(
            role="Senior Arxiv Literature Reviewer",
            goal="Conduct a comprehensive study of all research papers published on Arxiv on {}".format(self.user_input),
            backstory="You are an experienced research in the field of {} and holds a PhD. "
            "You have read and are well-informed about recent publications and research on {}.".format(self.user_input, self.user_input),
            tools=[arxiv_query_tool],
            max_iter=3,
            verbose=True,
            allow_delegation=False
        )

    def analyst(self):
        return Agent(
            role="Senior Techinal Research Analyst",
            goal="Write a technical report on {}.".format(self.user_input),
            backstory="You are a highly skilled te techinal research writer. You write highly insightful tech reports. " 
            "Include technical details specific to the {}. The post must provide some figures and facts to support your content. "
            "You take the output of Senior Web Researcher and Senior Arxiv Literature Reviewer and represent them in a more engaging "
            "and impactful way.".format(self.user_input),
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