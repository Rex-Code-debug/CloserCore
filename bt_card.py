import os
import operator
import json
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Annotated, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage
from urllib.parse import urljoin
from func import scrape_website, retriever_text, get_price_chunks
from langgraph.graph import StateGraph, END

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=os.environ.get("GROQ_API_KEY")
)

search_tool = DuckDuckGoSearchRun()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    company_name: str
    home_url: str
    description: str
    competitors: list[str]
    pricing_info: List[Dict[str, Any]]
    news_headlines: list
    loop_count: int
    final_report: str

def research_node(state: AgentState):
    """Search for company's official website, description, and competitors."""
    query = state["company_name"]
    count = state.get("loop_count", 0)

    print(f"🔍 Searching about {query}")

    website_url = "Not found"
    description = "Not available"
    competitors = []

    try:
        search_results = search_tool.invoke(f"{query} official website competitors")
        
        prompt = f"""From these search results, extract the official website URL, a 1-sentence description of what they do, and list of competitors for {query}.

Search results:
{search_results}

Return ONLY valid JSON in this exact format with no additional text, no markdown, no preamble:
{{
    "website_url": "https://example.com",
    "description": "One sentence description here",
    "competitors": ["competitor1", "competitor2", "competitor3"]
}}"""
        
        response = model.invoke(prompt)
        content = response.content.strip()
        
        # Clean markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            if content.startswith("json"):
                content = content[4:].strip()
        
        dat_js = json.loads(content)
        
        website_url = dat_js.get('website_url', 'Not found')
        description = dat_js.get('description', 'Not available')
        competitors = dat_js.get('competitors', [])
        
        print(f"✅ Found website: {website_url}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {str(e)}")
        print(f"Response content: {response.content if 'response' in locals() else 'No response'}")
        
    except Exception as e:
        print(f"❌ Error during search: {str(e)}")
    
    return {
        "home_url": website_url,
        "description": description,
        "competitors": competitors, 
        "loop_count": count + 1
    }

def pricing_node(state: AgentState):
    """Extract pricing information from pricing page"""
    pricing_info = state.get("pricing_info", [])
    home_url = state.get("home_url", "")
    
    if not home_url or home_url == "Not found":
        print("❌ No valid home URL found, skipping pricing extraction")
        return {"pricing_info": pricing_info}
    
    # Try multiple common pricing page URLs
    pricing_urls = [
        urljoin(home_url.rstrip('/') + '/', 'pricing'),
        urljoin(home_url.rstrip('/') + '/', 'plans'),
        urljoin(home_url.rstrip('/') + '/', 'price'),
    ]
    
    text = None
    successful_url = None
    
    for url in pricing_urls:
        try:
            print(f"💰 Trying pricing URL: {url}")
            text = scrape_website(url)
            if text and len(text) > 100:  # Check if we got meaningful content
                successful_url = url
                print(f"✅ Successfully fetched content from: {url}")
                break
        except Exception as e:
            print(f"⚠️ Failed to fetch {url}: {str(e)}")
            continue
    
    if not text:
        print("❌ No content retrieved from any pricing page")
        return {"pricing_info": pricing_info}

    try:
        chunks, embeddings = retriever_text(text)
        
        if chunks is None or embeddings is None:
            print("❌ Error: Could not create chunks or embeddings")
            return {"pricing_info": pricing_info}
        
        price_chunks = get_price_chunks(chunks, embeddings, top_k=3)
        combined_chunks = "\n\n---\n\n".join([chunk.page_content for chunk in price_chunks])

        prompt = f"""You are a pricing research expert. Analyze the following pricing information and extract specific details.

PRICING INFORMATION:
{combined_chunks}

TASK:
Extract pricing information and return ONLY a valid JSON object with no additional text, explanation, or markdown formatting.

RULES:
- For "free_tier": return true if there's a free plan (look for "free", "free forever", "$0"), otherwise false
- For "starter_plan": Find the CHEAPEST paid plan after free tier and include:
  * "name": the plan name (e.g., "Pro", "Basic", "Plus", "Starter")
  * "price": the price with currency (e.g., "$7.25/month", "₹500/month")
  * If not found, use null for both
- For "enterprise_plan": return true if enterprise/custom pricing exists (look for "Enterprise", "Custom", "Contact Sales"), otherwise false

IMPORTANT: Return ONLY the JSON object, no markdown code blocks, no explanations.

JSON FORMAT:
{{
    "free_tier": true,
    "starter_plan": {{
        "name": "plan name or null",
        "price": "price with currency or null"
    }},
    "enterprise_plan": true
}}
"""
        
        res = model.invoke(prompt)
        
        content = res.content.strip()
        
        # Clean markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            if content.startswith("json"):
                content = content[4:].strip()
        
        res_json = json.loads(content)
        pricing_info.append(res_json)
        print(f"✅ Successfully extracted pricing information")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in pricing_node: {str(e)}")
        print(f"Response content: {content if 'content' in locals() else 'No content'}")
        
    except Exception as e:
        print(f"❌ Error in pricing_node: {str(e)}")
    
    return {"pricing_info": pricing_info}

def news_node(state: AgentState):
    """Extract top 3 news headlines from blog page"""
    
    home_url = state.get('home_url', '')
    
    if not home_url or home_url == "Not found":
        print("❌ No valid home URL found, skipping news extraction")
        return {"news_headlines": []}
    
    # Try multiple common blog/news page URLs
    news_urls = [
        urljoin(home_url.rstrip('/') + '/', 'blog'),
        urljoin(home_url.rstrip('/') + '/', 'news'),
        urljoin(home_url.rstrip('/') + '/', 'press'),
        urljoin(home_url.rstrip('/') + '/', 'updates'),
    ]
    
    text = None
    successful_url = None
    
    for url in news_urls:
        try:
            print(f"📰 Trying news URL: {url}")
            res = requests.get(url=url, timeout=10)
            
            if res.status_code == 200:
                text = scrape_website(url)
                if text and len(text) > 100:  # Check if we got meaningful content
                    successful_url = url
                    print(f"✅ Successfully fetched content from: {url}")
                    break
        except Exception as e:
            print(f"⚠️ Failed to fetch {url}: {str(e)}")
            continue
    
    if not text:
        print("❌ No content retrieved from any news/blog page")
        return {"news_headlines": []}

    try:
        chunks, embeddings = retriever_text(text)
        
        if chunks is None or embeddings is None:
            print("❌ Error: Could not retrieve text or create chunks")
            return {"news_headlines": []}
        
        # Get top chunks with headline/news content
        chunks = get_price_chunks(chunks, embeddings, query="blog headline article title news", top_k=5)
        combined_chunks = "\n\n---\n\n".join([chunk.page_content for chunk in chunks])

        prompt = f"""You are a content analyst expert. Analyze the following blog/news content and extract the top 3 headlines.

CONTENT:
{combined_chunks}

TASK:
Extract the top 3 most prominent headlines or article titles from the content above.

RULES:
- Find actual headlines/titles of articles or blog posts
- Headlines should be clear, complete sentences or phrases
- Prioritize the most recent or featured articles
- Do not include navigation text, menu items, or generic labels
- Each headline should be meaningful and represent an actual article
- If you find fewer than 3 headlines, return only what you find

IMPORTANT: Return ONLY a valid JSON object with no additional text, explanation, or markdown formatting.

JSON FORMAT:
{{
  "headlines": [
    {{
      "title": "First headline text",
      "position": 1
    }},
    {{
      "title": "Second headline text",
      "position": 2
    }},
    {{
      "title": "Third headline text",
      "position": 3
    }}
  ]
}}

If fewer than 3 headlines are found, return only the available ones.
"""

        response = model.invoke(prompt)
        
        # Parse and clean the response
        content = response.content.strip()
        
        # Clean markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            if content.startswith("json"):
                content = content[4:].strip()
        
        # Validate JSON
        try:
            headlines_data = json.loads(content)
            print(f"✅ Successfully extracted {len(headlines_data.get('headlines', []))} headlines")
            return {"news_headlines": headlines_data.get('headlines', [])}
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON: {str(e)}")
            return {"news_headlines": []}
            
    except Exception as e:
        print(f"❌ Error in news_node: {str(e)}")
        return {"news_headlines": []}

def writer_node(state: AgentState):
    """Generate and save the final Battle Card report."""
    
    # Initialize the writer model with higher temperature for better writing
    model_writer = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    # Extract data from state
    company_name = state.get("company_name", "Unknown Company")
    description = state.get("description", "Not available")
    competitors = state.get("competitors", [])
    pricing_info = state.get("pricing_info", [])
    news_headlines = state.get("news_headlines", [])
    home_url = state.get("home_url", "Not found")
    
    # Format competitors
    competitors_text = ", ".join(competitors) if competitors else "Unknown"
    
    # Format pricing information
    if pricing_info and len(pricing_info) > 0:
        pricing_data = pricing_info[0]  # Get the first pricing info
        
        free_tier = "✅ Yes" if pricing_data.get("free_tier", False) else "❌ No"
        
        starter_plan = pricing_data.get("starter_plan", {})
        if starter_plan and starter_plan.get("name") and starter_plan.get("price"):
            starter_text = f"{starter_plan['name']}: {starter_plan['price']}"
        else:
            starter_text = "Unknown"
        
        enterprise = "✅ Available" if pricing_data.get("enterprise_plan", False) else "❌ Not Available"
        
        pricing_text = f"""
- **Free Tier:** {free_tier}
- **Starter Plan:** {starter_text}
- **Enterprise Plan:** {enterprise}
"""
    else:
        pricing_text = "Unknown"
    
    # Format news headlines
    if news_headlines and len(news_headlines) > 0:
        news_text = "\n".join([f"{i+1}. {headline.get('title', 'N/A')}" 
                               for i, headline in enumerate(news_headlines)])
    else:
        news_text = "No recent news available"
    
    # Create the prompt for the LLM
    prompt = f"""You are a Sales Assistant. Write a professional Battle Card for {company_name}.

INSTRUCTIONS:
- Use ONLY the data provided below
- Do NOT hallucinate or add information not present in the data
- Keep it concise and sales-focused
- Use markdown formatting for readability
- If any information is missing or "Unknown", clearly state it as such

DATA PROVIDED:
---
Company: {company_name}
Website: {home_url}
Description: {description}
Competitors: {competitors_text}

Pricing Information:
{pricing_text}

Recent News/Blog Headlines:
{news_text}

---

BATTLE CARD LAYOUT:
Create a battle card with these sections:

# Battle Card: {company_name}

## 1. What They Do
[Brief description of the company and their main offering]

## 2. Key Competitors
[List the main competitors]

## 3. Pricing Structure
[Detail the pricing information available]

## 4. Recent News & Updates
[List recent headlines or news]

## 5. Sales Strategy Tips
[Brief insights on how to position against competitors based on the data]

---

Write the complete battle card now:"""

    print(f"📝 Generating Battle Card for {company_name}...")
    
    try:
        # Generate the battle card using LLM
        response = model_writer.invoke(prompt)
        battle_card = response.content
        
        # Save to file
        filename = f"{company_name.replace(' ', '_').lower()}_battle_card.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(battle_card)
        
        print(f"✅ Battle Card saved to {filename}")
        
        # Also save raw data as JSON for reference
        raw_data = {
            "company_name": company_name,
            "website": home_url,
            "description": description,
            "competitors": competitors,
            "pricing_info": pricing_info,
            "news_headlines": news_headlines
        }
        
        json_filename = f"{company_name.replace(' ', '_').lower()}_data.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Raw data saved to {json_filename}")
        
        # Return updated state
        return {
            "final_report": battle_card,
            "loop_count": state.get("loop_count", 0) + 1
        }
        
    except Exception as e:
        print(f"❌ Error generating battle card: {str(e)}")
        error_message = f"# Error Generating Battle Card\n\nError: {str(e)}"
        
        return {
            "final_report": error_message,
            "loop_count": state.get("loop_count", 0) + 1
        }

# Build the graph
builder = StateGraph(AgentState)

builder.add_node("detective", research_node)
builder.add_node("pricing", pricing_node)
builder.add_node("news", news_node)
builder.add_node("editor", writer_node)

# Add edges to define the workflow
builder.set_entry_point("detective")
builder.add_edge("detective", "pricing")
builder.add_edge("pricing", "news")
builder.add_edge("news", "editor")
builder.add_edge("editor", END)

# Compile the graph
graph = builder.compile()

def run_battle_card_generator(company_name: str):
    """Run the battle card generator for a given company."""
    initial_state = {
        "messages": [],
        "company_name": company_name,
        "home_url": "",
        "description": "",
        "competitors": [],
        "pricing_info": [],
        "news_headlines": [],
        "loop_count": 0,
        "final_report": ""
    }
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Battle Card Generation for: {company_name}")
    print(f"{'='*60}\n")
    
    try:
        result = graph.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"✅ Battle Card Generation Complete!")
        print(f"{'='*60}\n")
        
        return result
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Test with a company name
    company_name = input("Enter company name: ")
    result = run_battle_card_generator(company_name)
    
    if result:
        print("\n📊 Final Report Preview:")
        print("-" * 60)
        print(result.get("final_report", "No report generated")[:500] + "...")