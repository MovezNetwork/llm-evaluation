import configparser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI
import pandas as pd

def get_evaluation_mistral(prompts_dict, input_data, output_name):
    output_evaluation_folder_path = 'f8_llm_evaluation_data/Mistral/'
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_key_mistral = config.get('credentials', 'api_key_mistral')
    mistral_client = MistralClient(api_key=api_key_mistral)
    mistral_m = "mistral-medium"

    scores_accuracy = []
    scores_content_preservation = []
    scores_fluency = []
    explanations_accuracy = []
    explanations_content_preservation = []
    explanations_fluency = []
    #for index, row in input_data.iterrows():

    # Accuracy
    query_accuracy = get_accuracy_prompt(prompts_dict, input_data)
    messages_accuracy = [ChatMessage(role="user", content=query_accuracy)]
    # No streaming
    chat_response_accuracy = mistral_client.chat(
        model=mistral_m,
        messages=messages_accuracy,
    )
    response_accuracy = chat_response_accuracy.choices[0].message.content
    #scores_accuracy.append(response_accuracy.split('\n')[0].split(' ')[1])
    explanations_accuracy.append(response_accuracy)

    # Content preservation
    query_content_preservation = get_content_preservation_prompt(prompts_dict, input_data)
    messages_content_preservation = [ChatMessage(role="user", content=query_content_preservation)]
    # No streaming
    chat_response_content_preservation = mistral_client.chat(
        model=mistral_m,
        messages=messages_content_preservation,
    )
    response_content_preservation = chat_response_content_preservation.choices[0].message.content
    #scores_content_preservation.append(response_content_preservation.split('\n')[0].split(' ')[1])
    explanations_content_preservation.append(response_content_preservation)

    # Fluency
    query_fluency = get_fluency_prompt(prompts_dict, input_data)
    messages_fluency = [ChatMessage(role="user", content=query_fluency)]
    # No streaming
    chat_response_fluency = mistral_client.chat(
        model=mistral_m,
        messages=messages_fluency,
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
    output.to_csv(output_evaluation_folder_path + "evaluation_" + output_name + '_mistral-medium.csv', index=False)

    return output



def get_updated_evaluation_mistral(prompts_dict, input_data, output_name,metrics):
    output_evaluation_folder_path = 'f8_llm_evaluation_data/Mistral/'
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_key_mistral = config.get('credentials', 'api_key_mistral')
    mistral_client = MistralClient(api_key=api_key_mistral)
    mistral_m = "mistral-medium"

    scores = []
    explanations = []

    score_column = ''
       
    query = get_updated_evaluation_prompt(prompts_dict, input_data,metrics)
    # print('prompt query: ',query,' \n')
    messages = [ChatMessage(role="user", content=query)]
    # No streaming
    chat_response = mistral_client.chat(
        model=mistral_m,
        messages=messages,
    )
    response = chat_response.choices[0].message.content
    # print('prompt response: ',response,' \n')
    if(metrics == 'feedback_generation'):
        scores.append('NaN')
        score_column = 'tst_feedback'
    if(metrics == 'feedback_evaluation'):
        scores.append(response.split('\n')[0].split(' ')[1])
        score_column = 'accuracy_feedback'
    else:
        scores.append(response.split('\n')[0].split(' ')[1])
        score_column = 'score_' + metrics
    explanations.append(response)

   

    output = pd.DataFrame(input_data).T
    output[score_column] = scores
    output['explanation_' + metrics] = explanations

    
    # Save output in a csv (locally)
    output.to_csv(output_evaluation_folder_path + "updated_evaluation_"+ metrics + '_' + output_name + '_mistral-medium.csv', index=False)

    return output


def get_updated_evaluation_prompt(prompts_dict, row, metrics):
    
    if metrics == 'feedback_evaluation':
        tst_text = row['tst_feedback']
    else:
        tst_text = row['rewritten_sentence']
    string_llm = f"{prompts_dict.get('prompt_llm').replace('{}', f'{{{tst_text}}}')}"

    # reads the user-based input examples - from 5 shot mistral
    df_m = pd.read_csv('f4_shots_data/' + row['user'] + '_mistral_shots_5.csv')
    user_style_string = '; '.join(df_m['original'])
    string_accuracy = f"{prompts_dict.get('prompt_s2').replace('{}', f'{{{user_style_string}}}')}"

    if metrics == 'feedback_generation':
        # use the explanation of the accuracy score for the feedback generation in the inference part of the string
        explanation_string = row['new_explanation_accuracy']
        string_inference = f"{prompts_dict.get('prompt_inference').replace('{}', f'{{{explanation_string}}}')}"
    else:
        string_inference = prompts_dict.get('prompt_inference')

    prompt = string_llm + string_accuracy + string_inference
    
    return prompt

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

    scores_accuracy = []
    scores_content_preservation = []
    scores_fluency = []
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
