from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

import math
import json
from datetime import datetime

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-2.5-flash"

# ---

def get_weather(city: str) -> dict:
    # Realistically, calling a weather API, not an LLM or somewhere else
    data = {
        "austin":   {"temp_f": 82, "condition": "sunny",  "humidity": 45},
        "new york": {"temp_f": 68, "condition": "cloudy", "humidity": 72},
        "seattle":  {"temp_f": 55, "condition": "rainy",  "humidity": 88},
        "miami":    {"temp_f": 91, "condition": "humid",  "humidity": 90},
        "chicago":  {"temp_f": 61, "condition": "windy",  "humidity": 60},
    }; key = city.lower()

    return {
        "status": "ok",
        "city": city,
        **data[key]
    } if key in data else {
        "status": "error",
        "message": f"No weather data for: {city}"
    }

def calculate(expression: str) -> dict:
    # Evaluates a math expression using Python's eval
    try:
        safe_math = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_math)
        return {"expression": expression, "result": round(float(result), 6), "status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_products(query: str, limit: int = 3) -> dict:
    # Searches for a product, realistically, in a database
    products = [
        {"id": 1, "name": "Wireless Headphones", "price": 89.99,  "category": "electronics"},
        {"id": 2, "name": "Running Shoes",        "price": 129.99, "category": "footwear"},
        {"id": 3, "name": "Coffee Maker",         "price": 59.99,  "category": "kitchen"},
        {"id": 4, "name": "Yoga Mat",             "price": 34.99,  "category": "fitness"},
        {"id": 5, "name": "Laptop Stand",         "price": 49.99,  "category": "office"},
        {"id": 6, "name": "Bluetooth Speaker",    "price": 69.99,  "category": "electronics"},
    ]; q = query.lower()
    matches = [p for p in products if q in p["name"].lower() or q in p["category"].lower()]

    return {
        "status": "ok",
        "results": matches[:limit],
        "count": len(matches)
    } if len(matches) > 0 else {
        "status": "error",
        "message": f"No results for: {query}"
    }

def get_current_time() -> dict:
    now = datetime.utcnow()
    return {"time": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d"), "status": "ok"}

# ---

tool_functions = {
    "get_weather":    get_weather,
    "calculate":      calculate,
    "search_products": search_products,
    "get_current_time": get_current_time,
}

def execute_tool(name: str, args: dict):
    print(f"Executing: {name}")
    fn = tool_functions.get(name)
    if fn is None:
        return { "status": "error", "message": f"No tool for: {name}" }
    
    return json.dumps(fn(**args))

# ---

def run_agent_auto(user_msg: str) -> str:
    # Automatic function call where the LLM handles the loop for us
    # With tools in the config, the SDK will read both code & docstrings for context to build JSON schemas

    res = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            tools=[get_weather, calculate, search_products, get_current_time],
            max_output_tokens=1024,
            system_instruction="""
                You are a helpful assistant with tools for weather, math,
                product search, and time. Use them whenever they help. Be concise.
            """
        ),
        contents=user_msg
    ); return res

# print(run_agent_auto("How's the weather in New York?"))

# ---

def run_agent_manual(user_msg: str) -> str:
    # Manual tool loop
    # 1. Send message to model, then returns a request for tool usage
    # 2. Execute that tool, get the result
    # 3. Send result back to model, the model may request for more tools or final answer
    # 4. Repeat until the reason isn't requesting for more tools

    contents=[types.Content(role="user", parts=[types.Part(text=user_msg)])]
    tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get current weather for a city",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"city": types.Schema(type=types.Type.STRING, description="City name")},
                    required=["city"]
                )
            ),
                        types.FunctionDeclaration(
                name="calculate",
                description="Evaluate a math expression. Supports sqrt, pow, sin, cos, log.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"expression": types.Schema(type=types.Type.STRING, description="Math expression")},
                    required=["expression"],
                ),
            ),
            types.FunctionDeclaration(
                name="search_products",
                description="Search the product database.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(type=types.Type.STRING, description="Search query"),
                        "limit": types.Schema(type=types.Type.INTEGER, description="Max results"),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_current_time",
                description="Get the current date and time.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={}),
            )
        ])
    ]; max_loops = 10

    for x in range(max_loops): # The model needs to loop to be able to reason one at a time based off of the OG user msg that include multiple tool requests
        # x = safety ceiling so the loop isn't infinite
        res = client.models.generate_content_stream(
            model=model,
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                system_instruction="""
                    You are a helpful assistant with tools for weather, math,
                    product search, and time. Use them whenever they help. Be concise.
                """
            ),
            contents=contents # Send array of contents, not user message
        )

        if not res.candidates: return "No response from model"

        candidate = res.candidates[0] # Is a list of possible responses to a single prompt
        # candidate > content >> parts, finish_reason, ...
        finish_reason = str(candidate.finish_reason) # A reason for why the model stopped
        # Enum of FinishReason: STOP, MAX_TOKENS, TOOL_CODE

        response_parts = candidate.content.parts
        func_calls = [p for p in response_parts if p.function_call is not None]
        # The functions requested by the model

        if not func_calls:
            final_res = "".join(p.text for p in response_parts if p.text)
            return final_res
        
        contents.append(types.Content(role="model", parts=response_parts))

        result_parts = []
        for part in func_calls:
            fc = part.function_call
            print(f"Tool requested: {fc}")

            func_result = execute_tool(fc.name, dict(fc.args))
            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": func_result}
                    )
                )
            )
        contents.append(types.Content(role="user", parts=result_parts))
        # role="user" because the tool result comes from the "outside world", not from the model itself

        # ex:
        # user:   "Search for electronics and tell me Seattle weather"
        # model:  [function_call: get_weather(city="Seattle")]
        # [function_call: search_products(query="electronics")]
        # user:   [function_response: get_weather → {temp: 55, ...}]
        # [function_response: search_products → {results: [...]}]
        # model:  "Seattle is 55°F and rainy. Here are some electronics..."

    return "Max loops reached"

# print(run_agent_manual("Search for electronics products and tell me the weather in Seattle"))
# Successfully found weather data for Seattle but not electronics since it searches by name and not by category