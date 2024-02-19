import configparser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI


def get_evaluation_mistral(prompts_dict, input_data, output_name):
    output_evaluation_folder_path = 'output_evaluation/'
    config = configparser.ConfigParser()
    config.read('config.ini')

    api_key_mistral = config.get('credentials', 'api_key_mistral')
    mistral_client = MistralClient(api_key=api_key_mistral)
    mistral_m = "mistral-medium"

    scores_accuracy = []
    scores_content_preservation = []
    scores_fluency = []
    for index, row in input_data.iterrows():

        # Accuracy
        query_accuracy = get_accuracy_prompt(prompts_dict, row)
        messages_accuracy = [ChatMessage(role="user", content=query_accuracy)]
        # No streaming
        chat_response_accuracy = mistral_client.chat(
            model=mistral_m,
            messages=messages_accuracy,
        )
        scores_accuracy.append(chat_response_accuracy.choices[0].message.content)

        # Content preservation
        query_content_preservation = get_content_preservation_prompt(prompts_dict, row)
        messages_content_preservation = [ChatMessage(role="user", content=query_content_preservation)]
        # No streaming
        chat_response_content_preservation = mistral_client.chat(
            model=mistral_m,
            messages=messages_content_preservation,
        )
        scores_content_preservation.append(chat_response_content_preservation.choices[0].message.content)

        # Fluency
        query_fluency = get_fluency_prompt(prompts_dict, row)
        messages_fluency = [ChatMessage(role="user", content=query_fluency)]
        # No streaming
        chat_response_fluency = mistral_client.chat(
            model=mistral_m,
            messages=messages_fluency,
        )
        scores_fluency.append(chat_response_fluency.choices[0].message.content)

    output = input_data
    output['score_accuracy'] = scores_accuracy
    output['score_content_preservation'] = scores_content_preservation
    output['score_fluency'] = scores_fluency

    # Save output in a csv (locally)
    output.to_csv(output_evaluation_folder_path + "evaluation_" + output_name + '_model_mistral_m.csv', index=False)

    return output


def get_evaluation_gpt(prompts_dict, input_data, output_name):
    output_evaluation_folder_path = 'output_evaluation/'
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
    for index, row in input_data.iterrows():
        # adapt prompts to input data
        string_llm = f"{prompts_dict.get('prompt_llm').replace('{}', f'{{{row[7]}}}')}"
        string_accuracy = f"{prompts_dict.get('prompt_tst_accuracy_s2').replace('{}', f'{{{row[8]}}}')}"
        string_content_preservation = f"{prompts_dict.get('prompt_content_preservation_s2').replace('{}', f'{{{row[6]}}}')}"

        # Accuracy
        query_accuracy = get_accuracy_prompt(prompts_dict, row)
        message_accuracy = [{"role": "user", "content": query_accuracy}]
        chat_response_accuracy = gtp_client.chat.completions.create(
            model=gpt_m,
            messages=message_accuracy,
            temperature=gpt_temperature,
            max_tokens=gpt_max_tokens,
            frequency_penalty=gpt_frequency_penalty
        )
        scores_accuracy.append(chat_response_accuracy.choices[0].message.content)

        # Content preservation
        query_content_preservation = get_content_preservation_prompt(prompts_dict, row)
        message_content_preservation = [{"role": "user", "content": query_content_preservation}]
        chat_response_content_preservation = gtp_client.chat.completions.create(
            model=gpt_m,
            messages=message_content_preservation,
            temperature=gpt_temperature,
            max_tokens=gpt_max_tokens,
            frequency_penalty=gpt_frequency_penalty
        )
        scores_content_preservation.append(chat_response_content_preservation.choices[0].message.content)

        # Fluency
        query_fluency = get_fluency_prompt(prompts_dict, row)
        message_fluency = [{"role": "user", "content": query_fluency}]
        chat_response_fluency = gtp_client.chat.completions.create(
            model=gpt_m,
            messages=message_fluency,
            temperature=gpt_temperature,
            max_tokens=gpt_max_tokens,
            frequency_penalty=gpt_frequency_penalty
        )
        scores_fluency.append(chat_response_fluency.choices[0].message.content)

    output = input_data
    output['score_accuracy'] = scores_accuracy
    output['score_content_preservation'] = scores_content_preservation
    output['score_fluency'] = scores_fluency

    # Save output in a csv (locally)
    output.to_csv(output_evaluation_folder_path + "evaluation_" + output_name + '_model_gpt_4.csv', index=False)

    return output


def get_accuracy_prompt(prompts_dict, row):
    string_llm = f"{prompts_dict.get('prompt_llm').replace('{}', f'{{{row[7]}}}')}"
    string_accuracy = f"{prompts_dict.get('prompt_accuracy_s2').replace('{}', f'{{{row[8]}}}')}"
    prompt = string_llm + string_accuracy + prompts_dict.get('prompt_accuracy_inference')
    return prompt


def get_content_preservation_prompt(prompts_dict, row):
    string_llm = f"{prompts_dict.get('prompt_llm').replace('{}', f'{{{row[7]}}}')}"
    string_content_preservation = f"{prompts_dict.get('prompt_content_preservation_s2').replace('{}', f'{{{row[6]}}}')}"
    prompt = string_llm + string_content_preservation + prompts_dict.get('prompt_content_preservation_inference')
    return prompt


def get_fluency_prompt(prompts_dict, row):
    string_llm = f"{prompts_dict.get('prompt_llm').replace('{}', f'{{{row[7]}}}')}"
    prompt = string_llm + prompts_dict.get('prompt_fluency_inference')
    return prompt
