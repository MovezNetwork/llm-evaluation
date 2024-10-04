import configparser
import json

from mistralai import Mistral
from openai import OpenAI
import pandas as pd
import random
import os

from tqdm import tqdm


def get_df_eval_acc (df_evaluation_accuracy, tst_output, examples):
    df_eval_acc = []

    for index, row in df_evaluation_accuracy.iterrows():
        message_id = row['message_id']
        userA = row['user_message_A']
        userB = row['user_message_B']
        row['message_A'] = tst_output.query('id_neutral_message == @message_id and username == @userA')['tst_message'].iloc[0]
        row['message_B'] = tst_output.query('id_neutral_message == @message_id and username == @userB')['tst_message'].iloc[0]

        data_by_user = examples.loc[examples['username'] == userA]
        row['examples'] = "\n".join(data_by_user['original'])

        df_eval_acc.append(row)

    return pd.DataFrame(df_eval_acc)


def tst_eval(prompt, input_data, model_name, shots):
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
        final_output = []
        query = prompt.format(row['examples'], row['message_A'], row['message_B'])

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
            messages = [{"role": "system", "content": ""}, {"role": "user", "content": query}]
            chat_response = gpt_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                response_format={'type': 'json_object'}
            )

        # Datetime information represented as string as a timestamp
        final_output.append({
            "model": chat_response.model,
            'shots': shots,
            'message_id': row['message_id'],
            'user_mA': row['user_message_A'],
            'user_mB': row['user_message_B'],
            'message_A': row['message_A'],
            'message_B': row['message_B'],
            'tst_eval': chat_response.choices[0].message.content,
            "query": query,
            "prompt_tokens": chat_response.usage.prompt_tokens,
            "completion_tokens": chat_response.usage.completion_tokens,
            "object": chat_response.object,
        })
        # final_output list to csv file
        df_output = pd.DataFrame(final_output)

        df_output_all = pd.concat([df_output_all, df_output], ignore_index=True)

    output_llm_folder_path = '03_tst_evaluation/'
    if not os.path.exists(output_llm_folder_path):
        os.makedirs(output_llm_folder_path)
    df_output_all.to_csv(output_llm_folder_path+ 'tst_eval_' + str(model_name) + '_' + str(shots) + '.csv', index=False)

    return df_output_all


def tst_eval_accuracy(prompt_system, prompt_user, input_data, model_name, shots, temp, seed):
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
        mistral_client = Mistral(api_key=api_key_mistral)

    elif 'gpt' in model_name:
        gpt_client = OpenAI(api_key=api_key_gpt)

    # create empty dataframe
    df_output_all = pd.DataFrame()

    # For each sentence, query Mistral API by looping over the neutral_sentences dataframe
    # use tqdm to show progress bar for the loop
    for index, row in tqdm(input_data.iterrows(), total=len(input_data), desc='Processing LLM TST evaluation...'):
        final_output = []
        query = prompt_user.format(row['original'], row['tst_message'])

        if 'mistral' in model_name:
            chat_response = mistral_client.chat.complete(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": prompt_system,
                        "role": "user",
                        "content": query,
                    },
                ],
                temperature=temp,
                random_seed=seed,
                response_format={'type': 'json_object'}
            )

        elif 'gpt' in model_name:
            messages = [{"role": "system", "content": ""}, {"role": "user", "content": query}]
            chat_response = gpt_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": prompt_system,
                        "role": "user",
                        "content": query,
                    },
                ],
                temperature=temp,
                seed=seed,
                response_format={'type': 'json_object'}
            )

        # Datetime information represented as string as a timestamp
        final_output.append({
            "model": chat_response.model,
            'temperature': temp,
            'shots': shots,
            'username': row['username'],
            'message_id': row['message_id'],
            'tst_message': row['tst_message'],
            'tst_eval': chat_response.choices[0].message.content,
            "query": query,
            "prompt_tokens": chat_response.usage.prompt_tokens,
            "completion_tokens": chat_response.usage.completion_tokens,
            "object": chat_response.object,
            'seed': seed,
        })

        # final_output list to csv file
        df_output = pd.DataFrame(final_output)

        df_output_all = pd.concat([df_output_all, df_output], ignore_index=True)

    return df_output_all




def parse_tst_data(df):
    wronglyParsed = 0
    for index, row in df.iterrows():
        try:
            data = fix_and_parse_json(row["tst_eval"])
            # Access the value
            score = data["answer"]
            explanation = data["explanation"]
            # Add the sentence to the DataFrame, in a new column called 'rewritten'
            df.loc[index, "score"] = score
            df.loc[index, "explanation"] = explanation
        except ValueError as e:
            print(e)
            df.loc[index, "eval_message"] = row["tst_eval"]
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
    if json_str.startswith('{{') and json_str.endswith('}}'):
        fixed_str = json_str[1:-1]  # Remove the outermost curly braces
        if is_valid_json(fixed_str):
            return json.loads(fixed_str)

    raise ValueError("The JSON string is not valid even after removing extra curly braces! Json_str: ", json_str)


def eval_summary(df):
    df.loc[:, 'score_num'] = df['score'].str.count('A')

    summary = df.groupby('user_mA').agg({
        'score_num': 'sum',        # Sum the counts of 'hello'
        'model': 'first',   # Keep the first location (same value for all rows in each group)
        'shots': 'first'      # Keep the first status (same value for all rows in each group)
    }).reset_index()

    return summary

def get_prompt_examples(prompt, examples):
    x_shots_list = []
    formatted_k_shot_string = ''

    for _, row in examples.iterrows():
        x_shots_list.append(row['original'])

    for i in range(0, len(x_shots_list),2):
        formatted_k_shot_string += prompt.format(x_shots_list[i]) + "\n"

    return formatted_k_shot_string
















def get_evaluation_mistral(prompt, input_messages, output_name):

    config = configparser.ConfigParser()
    config.read('config.ini')

    api_key_mistral = config.get('credentials', 'api_key_mistral')
    mistral_client = Mistral(api_key = api_key_mistral)
    model_name = 'mistral-large-latest'

    # create empty dataframe
    df_output_all = pd.DataFrame()

    for i, message in input_messages.iterrows():
        final_output = []
        input_message = message['tst_message']

        query = prompt
        query = f"{query.replace('{}', f'{{{input_message}}}')}"

        chat_response = mistral_client.chat.complete(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': query,
                },
            ],
            response_format={'type': 'json_object'}
        )

        # Datetime information represented as string as a timestamp
        final_output.append({
            'id_neutral_message': int(message['id_neutral_message']),
            'neutral_message': message['neutral_message'],
            'username': message['username'],
            'tst_id': message['tst_id'],
            'tst_message': message['tst_message'],
            'tst_evaluation': chat_response.choices[0].message.content,
            "query": query,
            "model": chat_response.model,
            "prompt_tokens": chat_response.usage.prompt_tokens,
            "completion_tokens": chat_response.usage.completion_tokens,
            "object": chat_response.object,
        })

        # final_output list to csv file
        df_output = pd.DataFrame(final_output)

        df_output_all = pd.concat([df_output_all, df_output], ignore_index=True)

    output_llm_folder_path = '03_tst_evaluation/'
    if not os.path.exists(output_llm_folder_path):
        os.makedirs(output_llm_folder_path)
    df_output.to_csv(output_llm_folder_path+output_name, index=False)

    return df_output_all




def get_evaluation_gpt(prompts_dict, input_data, output_name):
    output_evaluation_folder_path = 'f8_llm_evaluation_data/GPT/'
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_key_openai = config.get('credentials', 'api_key_openai')
    gtp_client = OpenAI(api_key=api_key_openai)
    gpt_m = "gpt-4"
    gpt_temperature = 0.2
    gpt_max_tokens = 256
    gpt_frequency_penalty = 0.0


    explanations_accuracy = []
    explanations_content_preservation = []
    explanations_fluency = []

    # Accuracy
    query_accuracy = get_accuracy_prompt(prompts_dict, input_data)
    message_accuracy = [{"role": "user", "content": query_accuracy}]
    chat_response_accuracy = gtp_client.chat.completions.create(
        model=gpt_m,
        messages=message_accuracy,
        temperature=gpt_temperature,
        max_tokens=gpt_max_tokens,
        frequency_penalty=gpt_frequency_penalty
    )
    response_accuracy = chat_response_accuracy.choices[0].message.content
    #scores_accuracy.append(response_accuracy.split('\n')[0].split(' ')[1])
    explanations_accuracy.append(response_accuracy)

    # Content preservation
    query_content_preservation = get_content_preservation_prompt(prompts_dict, input_data)
    message_content_preservation = [{"role": "user", "content": query_content_preservation}]
    chat_response_content_preservation = gtp_client.chat.completions.create(
        model=gpt_m,
        messages=message_content_preservation,
        temperature=gpt_temperature,
        max_tokens=gpt_max_tokens,
        frequency_penalty=gpt_frequency_penalty
    )
    response_content_preservation = chat_response_content_preservation.choices[0].message.content
    #scores_content_preservation.append(response_content_preservation.split('\n')[0].split(' ')[1])
    explanations_content_preservation.append(response_content_preservation)

    # Fluency
    query_fluency = get_fluency_prompt(prompts_dict, input_data)
    message_fluency = [{"role": "user", "content": query_fluency}]
    chat_response_fluency = gtp_client.chat.completions.create(
        model=gpt_m,
        messages=message_fluency,
        temperature=gpt_temperature,
        max_tokens=gpt_max_tokens,
        frequency_penalty=gpt_frequency_penalty
    )
    response_fluency = chat_response_fluency.choices[0].message.content
    #scores_fluency.append(response_fluency.split('\n')[0].split(' ')[1])
    explanations_fluency.append(response_fluency)

    output = pd.DataFrame(input_data).T
    #output['score_accuracy'] = scores_accuracy
    #output['score_content_preservation'] = scores_content_preservation
    #output['score_fluency'] = scores_fluency
    output['explanation_accuracy'] = explanations_accuracy
    output['explanation_content_preservation'] = explanations_content_preservation
    output['explanation_fluency'] = explanations_fluency

    # Save output in a csv (locally)
    output.to_csv(output_evaluation_folder_path + "evaluation_" + output_name + '_gpt-4.csv', index=False)

    return output






def get_accuracy_prompt(prompts_dict, row):
    string_llm = f"{prompts_dict.get('prompt_llm').replace('{}', f'{{{row.iloc[7]}}}')}"
    string_accuracy = f"{prompts_dict.get('prompt_accuracy_s2').replace('{}', f'{{{row.iloc[8]}}}')}"
    prompt = string_llm + string_accuracy + prompts_dict.get('prompt_accuracy_inference')
    return prompt

def get_content_preservation_prompt(prompts_dict, row):
    string_llm = f"{prompts_dict.get('prompt_llm').replace('{}', f'{{{row.iloc[7]}}}')}"
    string_content_preservation = f"{prompts_dict.get('prompt_content_preservation_s2').replace('{}', f'{{{row.iloc[6]}}}')}"
    prompt = string_llm + string_content_preservation + prompts_dict.get('prompt_content_preservation_inference')
    return prompt

def get_fluency_prompt(prompts_dict, row):
    string_llm = f"{prompts_dict.get('prompt_llm').replace('{}', f'{{{row.iloc[7]}}}')}"
    prompt = string_llm + prompts_dict.get('prompt_fluency_inference')
    return prompt

def get_accuracy_score(prompts_dict, input_data, output_name, loop_id, random_shots):
    output_evaluation_folder_path = 'f8_llm_evaluation_data/Mistral/'
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_key_mistral = config.get('credentials', 'api_key_mistral')
    mistral_client = MistralClient(api_key=api_key_mistral)
    mistral_m = "mistral-medium"

    scores = []
    explanations = []
       
    query = format_accuracy_score_prompt(prompts_dict, input_data, loop_id, random_shots)
    # print('Update Accuracy Prompt Query: ',query,' \n')

    messages = [ChatMessage(role="user", content=query)]
    # No streaming
    chat_response = mistral_client.chat(
        model=mistral_m,
        messages=messages,
    )
    response = chat_response.choices[0].message.content
    # print('Update Accuracy Prompt Response: ',response,' \n')

    scores.append(response.split('\n')[0].split(' ')[1])
    explanations.append(response)

   
    output = pd.DataFrame(input_data).T
    output['score_accuracy_'+ str(loop_id)] = scores
    output['explanation_accuracy_' + str(loop_id)] = explanations

    
    # Save output in a csv (locally)
    output.to_csv(output_evaluation_folder_path + 'loop_' + str(loop_id) + '_updated_accuracy_' + output_name + '_mistral-medium.csv', index=False)

    return output
    
def format_accuracy_score_prompt(prompts_dict, row, loop_id,random_shots):

    tst_text = row['tst_sentence_' + str(loop_id)]
    string_llm = f"{prompts_dict.get('prompt_p1').replace('{}', f'{{{tst_text}}}')}"

    # reads the user-based input examples - from 5 shot mistral
    if random_shots:
        df_random_shots = generate_five_shots_data(row['user'])
        user_style_string = '; '.join(df_random_shots['original'])
    else:
        df_static_shots = pd.read_csv('f4_shots_data/' + row['user'] + '_mistral_shots_5.csv')
        user_style_string = '; '.join(df_static_shots['original'])

    string_accuracy = f"{prompts_dict.get('prompt_p2').replace('{}', f'{{{user_style_string}}}')}"

    string_inference = prompts_dict.get('prompt_p3')

    prompt = string_llm + string_accuracy + string_inference
    
    return prompt



def generate_five_shots_data(user):
    
    user_data_file_name = 'f2_prompt_ready_chat_data/' + user + '_parallel_data_mistral_medium.csv'
    
    df_mistral = pd.read_csv(user_data_file_name)
    possible_rows = list(range(df_mistral.shape[0]))
    random.shuffle(possible_rows)
    five_shots = possible_rows[:5]

    return df_mistral.iloc[five_shots] 
             
        
            

def get_refinement_feedback(prompts_dict, input_data, output_name, loop_id):
    
    output_evaluation_folder_path = 'f8_llm_evaluation_data/Mistral/'
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key_mistral = config.get('credentials', 'api_key_mistral')
    mistral_client = MistralClient(api_key=api_key_mistral)
    mistral_m = "mistral-medium"
       
    query = format_refinement_feedback_prompt(prompts_dict, input_data, loop_id)
    print('Refinement Feedback Prompt Query: ',query,' \n')

    messages = [ChatMessage(role="user", content=query)]
    # No streaming
    chat_response = mistral_client.chat(
        model=mistral_m,
        messages=messages,
    )
    response = chat_response.choices[0].message.content
    print('Refinement Feedback Prompt Response: ',response,' \n')

    output = pd.DataFrame(input_data).T
    output['tst_sentence_' + str(loop_id + 1)] = ['NaN']
    output['explanation_tst_feedback_'  + str(loop_id)] = [response]

    # Save output in a csv (locally)
    output.to_csv(output_evaluation_folder_path + "refine_loop_" + str(loop_id) + "_refinement_feedback_" + '_' + output_name + '_mistral-medium.csv', index=False)

    return output


def format_refinement_feedback_prompt(prompts_dict, row, loop_id):

    tst_text = row['tst_sentence_' + str(loop_id)]
    string_llm = f"{prompts_dict.get('prompt_p1').replace('{}', f'{{{tst_text}}}')}"

    # reads the user-based input examples - from 5 shot mistral
    df_m = pd.read_csv('f4_shots_data/' + row['user'] + '_mistral_shots_5.csv')
    user_style_string = '; '.join(df_m['original'])
    string_accuracy = f"{prompts_dict.get('prompt_p2').replace('{}', f'{{{user_style_string}}}')}"

   
    # use the explanation of the accuracy score for the feedback generation in the inference part of the string
    if 'explanation_accuracy_' +  str(loop_id) in row.index:
        explanation_string = row['explanation_accuracy_' +  str(loop_id)]
    elif 'explanation_tst_feedback_' +  str(loop_id) in row.index:
         explanation_string = row['explanation_tst_feedback_' +  str(loop_id)]
   
    string_inference = f"{prompts_dict.get('prompt_p3').replace('{}', f'{{{explanation_string}}}')}"


    prompt = string_llm + string_accuracy + string_inference
    
    return prompt

def extract_explanation(df,column_name):
    ## Extracting the explanation text
    for i, value in df[column_name].items():
        if "Explanation:" in value:
            index = value.index("Explanation:") + len("Explanation:")
            value = value[index:]
            # print(i,value,'\n')
            df[column_name].loc[i] = value
            # print("Index found at:", index)
    return df
    
def extract_feedback(df,refine_loop_id):
   
    def remove_quotes(s):
        if s.startswith(' "'):
            s = s[2:]
        if s.endswith('"'):
            s = s[:-1]
        return s
        
    column = 'tst_sentence_' + str(refine_loop_id + 1)
    
    for i,r in df.iterrows():
        s = remove_quotes(r['explanation_tst_feedback_' + str(refine_loop_id)].split(':')[1].split('\n')[0])
        if s:
            df[column].iloc[i] = s 
        else:
            # some sentences are messed up, as Sentence: is followed with new line, so they require different handling 
            df[column].iloc[i] = r['explanation_tst_feedback_' + str(refine_loop_id)].split(':')[1].strip().split('\n')[0]

    return df