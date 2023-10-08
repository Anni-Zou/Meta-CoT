import time
import openai

# input your own openai api info below
openai.api_key = ""
openai.api_base = ""
openai.api_type = ""
openai.api_version = ""

def decoder_for_gpt(input, engine, max_length, role="", temperature=0):
    time.sleep(1)
    response = openai.ChatCompletion.create(
        engine=engine,
        messages=[
            {"role": "system", "content": "You are a brilliant assistant."},
            {"role": "user", "content":input}
        ],
        max_tokens=max_length,
        temperature=temperature
    )
    response = response['choices'][0]['message']['content']
    return response


def decoder_for_gpt_consistency(input, engine, max_length, n, temperature=0.7):
    time.sleep(1)
    responses = openai.ChatCompletion.create(
        engine=engine,
        messages=[
            {"role": "user", "content":input}
        ],
        max_tokens=max_length,
        temperature=temperature,
        n=n,
        stop=["\n"],
    )
    responses = [responses['choices'][i]['message']['content'] for i in range(n)]

    return responses
