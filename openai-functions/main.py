import logging
import time
import random
import openai
import json
import graphsignal

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(deployment='openai-functions-example')

# set OPENAI_API_KEY environment variable

# Parts of the example are taken from OpenAI's documentation
# https://platform.openai.com/docs/guides/gpt/function-calling

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

@graphsignal.trace_function
def run_conversation():
    # Step 1, send model the user query and what functions it has access to
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    message = response["choices"][0]["message"]

    # Step 2, check if the model wants to call a function
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        function_args = json.loads(message["function_call"]["arguments"])

        # Step 3, call the function
        # Note: the JSON response from the model may not be valid JSON
        with graphsignal.trace('get_current_weather') as span:
            function_response = get_current_weather(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            span.set_data('location', function_args.get("location"))
            span.set_data('response', function_response)

        # Step 4, send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": "What is the weather like in boston?"},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return second_response

while True:
    try:
        print(run_conversation())
    except:
        logger.error("Error running conversation", exc_info=True)

    time.sleep(20 * random.random())
