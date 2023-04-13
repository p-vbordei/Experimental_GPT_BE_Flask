from rich.console import Console
from rich.table import Table
console = Console()



from langchain.tools import BaseTool
import urlsummary as urls

class UrlSummary(BaseTool):
    name = "Get URL Summary"
    description = '''
    A tool that can get a brief summary of a URL
    
    Input: URL
    '''

    def _run(self, tool_input: str) -> str:
        console.log(f"[bold green]call {self.name}[/]: {tool_input}")
        try:
            '''
            a = Article(tool_input)
            a.download()
            a.parse()
            a.nlp()
            return a.summary
            '''
            r = urls.get_webpage_summary(tool_input)
            console.log(f"[bold green]return:[/]\n {r}")
            return r
        except Exception as e:
            return "Summary failed"

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError("UrlSummary does not support async")
    


# I want to do a summary to the URL
URL = '....'
url_summary = UrlSummary()
url_summary(URL)


# De testat mai mult si pe alte sites