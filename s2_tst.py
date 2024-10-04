import pandas as pd
import configparser
from mistralai import Mistral
from openai import OpenAI
from tqdm import tqdm
import datetime
import json
import random
import os
import glob

os.environ["WANDB_SILENT"] = "true"
from wandb.sdk.data_types.trace_tree import Trace
import wandb
import re


def llm_tst(df_user_data, neutral_messages, model_name, df_prompts, shots, temp, seed):
    config = configparser.ConfigParser()
    # Read the configuration file & paths
    config.read('config.ini')
    api_key_mistral = config.get('credentials', 'api_key_mistral')
    api_key_gpt = config.get('credentials', 'api_key_openai')

    # prompt types: 0 = parallel data, else = non-parallel data
    prompt_id = 0

    # 'open-mistral-nemo', 'mistral-large-latest','gpt-4o-mini','gpt-4o'
    #model_name = 'gpt-4o'

    system_prompt = df_prompts['prompt_system_content'].iloc[0]
    task_prompt = df_prompts['prompt_x_shot_template'].iloc[0]
    inference_prompt = df_prompts['prompt_inference'].iloc[0]

    # create empty dataframe
    df_output_all = pd.DataFrame()

    # For each user, generate the prompt and query Mistral API
    grouped_data = df_user_data.groupby('username')
    for username, group in tqdm(grouped_data, total=df_user_data['username'].nunique(),
                                desc='Generating LLM TST Messages per Participant...'):

        # for username, group in grouped_data:
        x_shots_list = []
        messages_id = []

        formatted_k_shot_string = ''

        # parallel data case
        if prompt_id == 0:
            for _, row in group.iterrows():
                x_shots_list.append(row['neutral'])
                x_shots_list.append(row['original'])
                messages_id.append(row['messageID'])

            for i in range(0, len(x_shots_list),2):
                formatted_k_shot_string += task_prompt.format(x_shots_list[i], x_shots_list[i + 1]) + "\n\n"
        # non-parallel data case
        else:
            for _, row in group.iterrows():
                # Access values in the desired order and append to the list
                x_shots_list.append(row['original'])
                messages_id.append(row['messageID'])
            for i in range(0, len(x_shots_list)):
                formatted_k_shot_string += task_prompt.format(x_shots_list[i]) + "\n\n"

        # Query Mistral API
        if 'mistral' in model_name:
            mistral_client = Mistral(api_key = api_key_mistral)

        elif 'gpt' in model_name:
            gpt_client = OpenAI(api_key = api_key_gpt)

        # For each sentence, query Mistral API by looping over the neutral_sentences dataframe
        # use tqdm to show progress bar for the loop
        for i, message in neutral_messages.iterrows():
            final_output = []
            neutral_message = message['message']

            query = formatted_k_shot_string + '\n' + inference_prompt
            query = f"{query.replace('{}', f'{{{neutral_message}}}')}"
            if 'mistral' in model_name:
                chat_response = mistral_client.chat.complete(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        }, {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=temp,
                    random_seed=seed,
                    response_format={'type': 'json_object'}

                )

            elif 'gpt' in model_name:
                chat_response = gpt_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                         }, {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=temp,
                    seed=seed,
                    response_format={'type': 'json_object'}
                )

            # Datetime information represented as string as a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            final_output.append({
                'message_id': int(message['messageID']),
                'neutral_message': message['message'],
                'username': username,
                'tst_id': username + timestamp,
                'llm_tst': chat_response.choices[0].message.content,
                "query": query,
                "model": chat_response.model,
                "prompt_tokens": chat_response.usage.prompt_tokens,
                "completion_tokens": chat_response.usage.completion_tokens,
                "object": chat_response.object,
                "promptID": str(prompt_id),
                'shots': shots,
                'temperature': temp,
                'seed': seed,
                "timestamp": timestamp,
            })
            # final_output list to csv file
            df_output = pd.DataFrame(final_output)

            #output_llm_folder_path = '02_tst_output/' + model_name + '/individual/'
            #if not os.path.exists(output_llm_folder_path):
            #    os.makedirs(output_llm_folder_path)
            #df_output.to_csv(output_llm_folder_path+'_u'+username+'_s'+str(message['messageID'])+'_t'+timestamp+'_'+shots+'_shots.csv', index=False)

            df_output_all = pd.concat([df_output_all, df_output], ignore_index=True)

    return df_output_all


def parse_tst_data(df):
    wronglyParsed = 0
    for index, row in df.iterrows():
        print(index)
        try:
            data = fix_and_parse_json(row["llm_tst"])
            print(data)
            # Access the value
            message = data["rewritten_sentence"]
            explanation = data["explanation"]
            # Add the sentence to the DataFrame, in a new column called 'rewritten'
            df.loc[index, "tst_message"] = message
            df.loc[index, "tst_explanation"] = explanation
        except ValueError as e:
            print(e)
            df.loc[index, "llm_message"] = row["llm_tst"]
            wronglyParsed += 1
    print("wronglyParsed: ", wronglyParsed)

    return df


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
    if json_str.startswith('{{'):
        fixed_str = json_str.replace("{", "", 1)  # Remove the first curly braces
        if is_valid_json(fixed_str):
            return json.loads(fixed_str)

    # Remove extra curly braces and try parsing again
    if json_str.endswith('}}'):
        fixed_str = json_str[::-1].replace("}", "", 1)[::-1]  # Remove the last curly braces
        if is_valid_json(fixed_str):
            return json.loads(fixed_str)

    raise ValueError("The JSON string is not valid even after removing extra curly braces! Json_str: ", json_str)


def has_nested_curly_braces(s):
    # Initialize a counter to track depth of curly braces
    depth = 0
    for char in s:
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
        # Check if there is a nested structure
        if depth > 1:
            return True
    return False


def fix_output(df):
    fixed = 0
    for index, row in df.iterrows():
        if has_nested_curly_braces(row['llm_tst']):
            result = re.sub(r'\{([^{}]*)\}', r'\1', row['llm_tst'])
            df.loc[index, 'llm_tst'] = result
            fixed+=1
    print('fixed: ', fixed)

    return df


def llm_tst_exploration(input_data, neutral_messages, model_name, prompt, shots):
    config = configparser.ConfigParser()
    # Read the configuration file & paths
    config.read('config.ini')
    api_key_mistral = config.get('credentials', 'api_key_mistral')
    api_key_gpt = config.get('credentials', 'api_key_openai')

    # prompt types: 0 = parallel data, else = non-parallel data
    prompt_id = 0

    # 'open-mistral-nemo', 'mistral-large-latest','gpt-4o-mini','gpt-4o'
    # Query Mistral API
    if 'mistral' in model_name:
        mistral_client = Mistral(api_key = api_key_mistral)

    elif 'gpt' in model_name:
        gpt_client = OpenAI(api_key = api_key_gpt)

    # create empty dataframe
    df_output_all = pd.DataFrame()

    # For each sentence, query Mistral API by looping over the neutral_sentences dataframe
    # use tqdm to show progress bar for the loop
    for index, row in tqdm(input_data.iterrows(), total=len(input_data), desc='Processing LLM TST evaluation...'):

        username = row['username']

        for i, message in neutral_messages.iterrows():
            final_output = []
            neutral_message = message['message']

            query = prompt.format(row['examples_original'], row['examples_neutral'], neutral_message)
            if 'mistral' in model_name:

                chat_response = mistral_client.chat.complete(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": query,
                        },
                    ],
                    response_format={'type': 'json_object'}
                )

            elif 'gpt' in model_name:
                chat_response = gpt_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=0.2,
                    response_format={'type': 'json_object'}
                )

            # Datetime information represented as string as a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            final_output.append({
                'message_id': int(message['messageID']),
                'neutral_message': message['message'],
                'username': username,
                'tst_id': username + timestamp,
                'llm_tst': chat_response.choices[0].message.content,
                "query": query,
                "model": chat_response.model,
                "prompt_tokens": chat_response.usage.prompt_tokens,
                "completion_tokens": chat_response.usage.completion_tokens,
                "object": chat_response.object,
                "promptID": str(prompt_id),
                'shots': shots,
                "timestamp": timestamp,
            })
            # final_output list to csv file
            df_output = pd.DataFrame(final_output)

            df_output_all = pd.concat([df_output_all, df_output], ignore_index=True)

    return df_output_all


def get_df_tst(examples):
    grouped_data = examples.groupby('username')
    df_eval_acc = []
    for username, group in tqdm(grouped_data, total=examples['username'].nunique()):
        row_data = {
            'username': username,
            'examples_original': "\n".join(group['original']),
            'examples_neutral': "\n".join(group['neutral'])
        }
        df_eval_acc.append(row_data)

    return pd.DataFrame(df_eval_acc)

