import pandas as pd
import configparser
import json

from mistralai import Mistral
from openai import OpenAI
from tqdm import tqdm


def create_parallel_corpus(df, model_name):
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.ini')
    api_key_mistral = config.get('credentials', 'api_key_mistral')
    api_key_gpt = config.get('credentials', 'api_key_openai')
    prompt_id = '111'
    prompt_content = '''
    You are an expert in text style transfer. 
    You will be given a sentence written in the conversational style of person X. 
    Your task is to rewrite the same sentence without any style. 
    Here is the sentence written in the style of person X: {}.
    Format result in json as {rewrittenSentence: ""}
    Do NOT provide any additional information or explanation. 
    
    '''

    if 'mistral' in model_name:
        mistral_client = Mistral(api_key = api_key_mistral)
    elif 'gpt' in model_name:
        gpt_client = OpenAI(api_key = api_key_gpt)

    df['parallelSentence'] = None
    final_output = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating Parallel Corpus..."):
        original = row['content']
        query = f"{prompt_content.replace('{}', f'{{{original}}}',1)}"

        if 'mistral' in model_name:
            chat_response = mistral_client.chat.complete(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                response_format = {
                    "type": "json_object",
                }
            )
        elif 'gpt' in model_name:
            messages = [{"role": "system", "content": 'You are an linguistics expert.'}, {"role": "user", "content": query}]
            chat_response = gpt_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )

        final_output.append({"timestamp": row['timestamp'],
                             "original": original,
                             "output": chat_response.choices[0].message.content,
                             "model": chat_response.model,
                             "prompt_tokens": chat_response.usage.prompt_tokens,
                             "completion_tokens": chat_response.usage.completion_tokens,
                             "object": chat_response.object, "promptID": prompt_id})

    df_output = pd.DataFrame(final_output)
    df_output = df_output.reset_index(names='messageID')
    # Apply the function to each row and create new columns
    df_output.to_csv('01_processed_input_data/parallel_data_'+model_name+'.csv')

    return df_output

def parse_parallel_data(df_par_sent):
    wronglyParsed = 0
    for index, row in df_par_sent.iterrows():
        try:
            data = fix_and_parse_json(row['output'])
            # Access the value
            sentence = data["rewrittenSentence"]
            # Add the sentence to the DataFrame, in a new column called 'rewritten'
            df_par_sent.loc[index, 'rewritten'] = sentence
        except ValueError as e:
            print(e)
            df_par_sent.loc[index, 'rewritten'] = row['output']
            wronglyParsed += 1
    print('wronglyParsed: ',wronglyParsed)

    return df_par_sent

def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False

def fix_and_parse_json(json_str):
    # Check if the JSON is already valid
    if is_valid_json(json_str):
        return json.loads(json_str)

    # Remove extra curly braces and try parsing again
    if json_str.startswith('{{') and json_str.endswith('}}'):
        fixed_str = json_str[1:-1]  # Remove the outermost curly braces
        if is_valid_json(fixed_str):
            return json.loads(fixed_str)

    raise ValueError("The JSON string is not valid even after removing extra curly braces! Json_str: ", json_str)

def merge_parallel_data(df_chats, df_par):
    df_par = df_par[['timestamp', 'rewritten']]
    df_chats = pd.merge(df_par, df_chats, on='timestamp')
    return df_chats

def get_k_shots(df,k, model):
    # Get random k messages from each participant
    df_k_shots = df.groupby('sessionId').apply(lambda x: x.sample(k)).reset_index(drop=True)
    df_k_shots = df_k_shots[['sessionId','messageID','timestamp','content','rewritten']]
    # rename content -> original, rewritten -> neutral
    df_k_shots = df_k_shots.rename(columns={'content':'original','rewritten':'neutral','sessionId':'username'})
    df_k_shots.to_csv('01_processed_input_data/'+str(k)+'_shots_data_'+model+'.csv')
    return df_k_shots

def get_all_shots(df,model):
    # Get all messages from each participant
    df_all_shots = df[['sessionId','messageID','timestamp','content','rewritten']]
    # rename content -> original, rewritten -> neutral
    df_all_shots = df_all_shots.rename(columns={'content':'original','rewritten':'neutral','sessionId':'username'})
    df_all_shots.to_csv('01_processed_input_data/all_shots_data_'+model+'.csv')
    return df_all_shots