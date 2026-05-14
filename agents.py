from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv

load_dotenv()

#model setup 
llm = ChatMistralAI(model = "mistral-medium-3-5", temperature=0) #USE ANY MODEL AVAILABLE


#1st agent 
def build_search_agent():
    return create_agent (
        model = llm,
        tools= [web_search],
        system_prompt="""
You are a search agent.

STRICT RULES:
- You MUST use the web_search tool.
- NEVER answer from your own knowledge.
- DO NOT summarize.
- RETURN the tool output EXACTLY as it is.
- Do not add explanations.
"""
    )
    

#2nd agent 

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are a reader agent.

STRICT RULES:
- You MUST use the scrape_url tool.
- Pass exactly one URL string to scrape_url, not commentary.
- If the best source is a PDF URL, still pass that PDF URL to scrape_url.
- DO NOT add extra explanations.
- RETURN the scraped content only.
"""
    )

#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert research writer. Write clear, structured and insightful reports.\n\n"
     "EVIDENCE DISCIPLINE:\n"
     "1. Use ONLY the evidence provided in Research Gathered and Relevant Past Research.\n"
     "2. Do NOT rely on general model knowledge for factual claims, even when the topic is familiar.\n"
     "3. Every Key Finding must be grounded in the provided evidence. If the evidence is thin, say that clearly.\n"
     "4. When Relevant Past Research is provided and it is topically related, actively incorporate its concrete facts in the report.\n"
     "5. Do not add names, dates, numbers, examples, or claims unless they appear in the provided evidence.\n\n"
     "STRICT RULES about the 'Connections to Past Research' section:\n"
     "1. First, CHECK if the past research context below is TOPICALLY RELATED to the current topic.\n"
     "2. If it says 'No relevant past research available' then write 'No prior research available for this topic.' and NOTHING else in that section.\n"
     "3. If past research IS provided but is about a DIFFERENT topic (e.g., oil prices when writing about water scarcity) then "
     "write 'Past research in the knowledge base covers different topics and is not directly applicable to this report.' Do NOT force connections.\n"
     "4. ONLY write substantive connections if the past research is genuinely about the SAME subject matter.\n"
     "5. NEVER fabricate, hallucinate, or invent connections. NEVER say 'builds on prior research' unless it actually does."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Relevant Past Research (from Knowledge Base):
{rag_context}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Connections to Past Research (follow the STRICT RULES above. If no relevant past research or if the past research is about unrelated topics, say so honestly and move on)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional, but stay inside the provided evidence. Do NOT invent past research connections that do not exist."""),
])


writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
