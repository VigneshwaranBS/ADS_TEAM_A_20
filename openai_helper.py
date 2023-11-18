import openai
from secret_key import openai_key
import json
import pandas as pd

openai.api_key = openai_key

def extract_financial_data(user_input_text):
    prompt = get_prompt_financial(user_input_text)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0]['message']['content']

    try:
        data = json.loads(content)
        if not data:
            return pd.DataFrame({
                "Message": ["No information found"],
                "Value": [""]
            })

        return pd.DataFrame(data.items(), columns=["Measure", "Value"])

    except (json.JSONDecodeError, IndexError):
        pass

    return pd.DataFrame({
        "Message": ["No information found"],
        "Value": [""]
    })

def get_prompt_financial(user_input_text):
    return f'''Please extract financial details from the following user input text:
    "{user_input_text}"
    
    If the exact data is not available from the user input, then use OpenAI's API to retrieve relevant information.
    
    Return your response as a valid JSON string. The format of that string should be similar to this,
    {{
        "Measure1": "Value1",
        "Measure2": "Value2",
        ...
    }}
    '''

if __name__ == '__main__':
    user_input_text = '''
    Tesla's Earning news in text format: Tesla's earning this quarter blew all the estimates. They reported 4.5 billion $ profit against a revenue of 30 billion $. Their earnings per share was 2.3 $
    '''
    df = extract_financial_data(user_input_text)

    print(df.to_string(index=False))
