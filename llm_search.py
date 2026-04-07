import os
import requests
import urllib.parse
import xml.etree.ElementTree as ET

def research_deepfake_query(query: str) -> str:
    """
    Takes a user query, fetches recent news via Google News RSS,
    and feeds them into OpenAI API (GPT-4o) to generate a comprehensive, RAG-style response.
    """
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        return "Ralat: Sila masukkan OPENAI_API_KEY ke dalam fail .env."

    context = ""
    
    # 1. Fetch web context using Google News RSS API (Free, No Keys, No Rate Limits)
    try:
        encoded_query = urllib.parse.quote(f"{query} deepfake")
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        rss_response = requests.get(rss_url, timeout=10)
        root = ET.fromstring(rss_response.text)
        items = root.findall("./channel/item")[:3]
        
        if items:
            results = []
            for item in items:
                title = item.find("title").text if item.find("title") is not None else "No Title"
                link = item.find("link").text if item.find("link") is not None else "No Link"
                results.append(f"Title: {title}\nLink: {link}\n")
            context = "\n".join(results)
        else:
            context = "Tiada konteks carian web, gunakan pengetahuan am terbina dalam GPT-4o."
    except Exception as e:
        print(f"Google News RSS Search failed: {e}")
        context = "Live web search failed. Answer using your internal base knowledge as best as you can."

    # 2. Build the prompt
    system_prompt = f"""You are TrueSight AI, an expert deepfake detection research assistant built to help users learn about media manipulation.
Use the following real-time web search results (if any) to answer the user's query about deepfakes comprehensively. 
If articles are provided in the context, you MUST include their URL Links in your response so the user can read the full article.
If the search results are empty or do not contain enough info, you must rely on your general knowledge.
Translate your response to conversational fluent Malay if the user asks in Malay, otherwise use English.
Always be professional, objective, and format your response nicely with markdown (bullet points, bold text, [Link Text](URL)).

== Real-Time Web Context ==
{context}

== User Query ==
{query}
"""
    
    # 3. Request AI Response from OpenAI GPT-4o
    try:
        print(f"Requesting OpenAI interpretation for: '{query}'")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            ai_response = data['choices'][0]['message']['content']
            return ai_response
        else:
            return f"Maaf, ralat OpenAI API: {response.status_code} - {response.text}"
            
    except Exception as e:
        print(f"Error in OpenAI Chat: {e}")
        return "Maaf, perkhidmatan AI (OpenAI) sedang mengalami masalah sambungan Rangkaian. Sila cuba lagi."
