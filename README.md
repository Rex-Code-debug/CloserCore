# 🎯 CloserCore | Agentic Competitive Intelligence Engine

**CloserCore** is a high-performance, multi-agent research system designed to automate competitive intelligence. Built with **LangGraph** and **Python**, it transforms a simple company name into a comprehensive "Battle Card" used by sales teams to outmaneuver (outperform) competitors and close deals faster.

## 🚀 Key Features
* **Autonomous Detective Node:** Automatically identifies official websites, company descriptions, and key competitors.
* **Deep-Dive Pricing Extraction:** Scrapes and parses complex pricing structures using intelligent chunking and RAG.
* **Sentiment & News Analysis:** Leverages DuckDuckGo search to find recent headlines and customer sentiment.
* **Stateful Orchestration:** Built on **LangGraph StateGraphs** to ensure fault-tolerant (error-resistant) execution and complex looping logic.
* **Markdown Generation:** Outputs professional, ready-to-use Battle Cards for Sales Ops.

## 🛠️ Technical Stack
* **Framework:** LangGraph (StateGraph)
* **Orchestration:** LangChain
* **LLM:** Llama-3.1-8b (via Groq) for ultra-fast inference
* **Data Acquisition:** BeautifulSoup4, Requests, DuckDuckGo Search
* **Environment:** Python 3.10+, Dotenv

## 📊 Logic Flow
CloserCore doesn't just run a script; it manages a stateful workflow:
1. **Research Phase** -> 2. **Pricing Extraction** -> 3. **Intelligence Gathering** -> 4. **Strategic Synthesis (LLM)** -> 5. **Final Delivery**
