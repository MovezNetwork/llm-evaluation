import datetime
import pandas as pd
import configparser
from mistralai import Mistral
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
import datetime
import re
import os
from stqdm import stqdm
from tqdm import tqdm
from openai import OpenAI

# Data preparation methods
def read_input_data():
    # EXPERIMENT 1
    # output_shots_data = 'f4_shots_data/'
    # csv_files = glob.glob(output_shots_data + '*.csv')

    # config = configparser.ConfigParser()
    # # Read the configuration file & paths
    # config.read('config.ini')

    # #define empty dataframe to store all shots
    # df_all_shots = pd.DataFrame()
    # for file in csv_files:
    #     if (file.find('mistral') != -1 and file.endswith('5.csv')):
    #             username = file[14:16]
    #             df_shots = pd.read_csv(file)
    #             # add username column to the df_all_shots and the df_shots dataframes
    #             df_shots['username'] = username
    #             df_all_shots = pd.concat([df_all_shots,df_shots[['username','messageID','original']]], ignore_index=True)

    # # Sort by username
    # df_all_shots = df_all_shots.sort_values(by='username')

    # surfdrive_url_input_sentences = config.get('credentials', 'surfdrive_url_input_sentences')
    # neutral_sentences = pd.read_csv(surfdrive_url_input_sentences,sep=';')[['idSentence','sentences']][0:10]
    # surfdrive_url_transcript_sentences = config.get('credentials', 'surfdrive_url_transcript_sentences')
    # user_sentences = pd.read_csv(surfdrive_url_transcript_sentences).reset_index()[['user', 'original', 'your_text']]
    # user_sentences = user_sentences.merge(neutral_sentences, left_on='original', right_on='sentences', how='left')
    # user_sentences = user_sentences.drop(columns=['sentences'])

    # EXPERIMENT 2
    config = configparser.ConfigParser()
    # Read the configuration file & paths
    config.read('config.ini')
    data_file = 'f1_processed_user_chat_data/five_shots.csv'
    df_all_shots = pd.read_csv(data_file)
    df_all_shots = df_all_shots.sort_values(by='username')

    surfdrive_url_input_sentences = config.get('credentials', 'surfdrive_url_input_sentences')
    neutral_sentences = pd.read_csv(surfdrive_url_input_sentences,sep=';')[['sentenceid','sentences']]


    surfdrive_url_transcript_sentences = config.get('credentials', 'surfdrive_url_between_us_user_rewritten')
    user_sentences = pd.read_csv(surfdrive_url_transcript_sentences,sep=';')[['userid','sentenceid','rewritten']]
    user_sentences = user_sentences.merge(neutral_sentences, on='sentenceid', how='left')

    return df_all_shots,neutral_sentences,user_sentences

# TST methods
def llm_tst(df_user_data, neutral_sentences,model_name, system_prompt, kshot_prompt, inference_prompt,prompt_id):
    df_output_all = pd.DataFrame()
    config = configparser.ConfigParser()
    # Read the configuration file & paths
    config.read('config.ini')
    api_key_mistral = config.get('credentials', 'api_key_mistral')
    api_key_gpt = config.get('credentials', 'api_key_openai')


    # create a new folder programmically, with name being a current timestamp in the format YYYYMMDDHHMMSS
    output_run = 'run_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '_model_' + model_name + '_type_' + str(prompt_id)
    # create prompt string and save it as txt
    prompt  = system_prompt + ' \n  ' +  kshot_prompt + ' \n  ' +  inference_prompt


    output_llm_folder_path = 'f6_llm_tst_data/' + output_run + '/'
    # create the folder if it does not exist
    if not os.path.exists(output_llm_folder_path):
        os.makedirs(output_llm_folder_path)

    # Specify the file name
    file_name = 'f6_llm_tst_data/' + output_run + '/'+  "prompt_" +output_run+ ".txt"
    with open(file_name, "w") as file:
        file.write(prompt)

    # For each user, generate the prompt and query Mistral API
    grouped_data = df_user_data.groupby('username')
    for username, group in stqdm(grouped_data,total=df_user_data['username'].nunique(),desc = "Generating LLM TST Sentences per User "):
    # for username, group in grouped_data:

        x_shots_list = []
        messages_id = []

        formatted_k_shot_string = ''
        

        # parallel data case
        if prompt_id==0:
            for _, row in group.iterrows():
                x_shots_list.append(row['neutral'])
                x_shots_list.append(row['original'])
                messages_id.append(row['messageID'])

            for i in range(0, len(x_shots_list),2):
                formatted_k_shot_string += kshot_prompt.format(x_shots_list[i], x_shots_list[i + 1]) + "\n\n"
        # non-parallel data case
        else:
            for _, row in group.iterrows():
                # Access values in the desired order and append to the list
                x_shots_list.append(row['original'])  
                messages_id.append(row['messageID'])        
            for i in range(0, len(x_shots_list)):
                formatted_k_shot_string += kshot_prompt.format(x_shots_list[i]) + "\n\n"
                
        
        # Query Mistral API
        if 'mistral' in model_name:
            # mistral_client = MistralClient(api_key = api_key_mistral)
            mistral_client = Mistral(api_key = api_key_mistral)

        elif 'gpt' in model_name:
            gpt_client = OpenAI(api_key = api_key_gpt)
        # For each sentence, query Mistral API by looping over the neutral_sentences dataframe
        # use tqdm to show progress bar for the loop
    

        for i, sentence in neutral_sentences.iterrows():
            final_output = []
            neutral_sentence = sentence['sentences']

            query = ''
            if 'mistral' in model_name:
                mistral_query = system_prompt + '\n' + formatted_k_shot_string + '\n' + inference_prompt
                mistral_query = f"{mistral_query.replace('{}', f'{{{neutral_sentence}}}')}"
                query = mistral_query
                # print('\n Query: ', query)

                # messages = [ChatMessage(role="user", content=query)]
                # chat_response = mistral_client.chat(
                #     model = model_name,
                #     messages = messages,
                # )

                chat_response = mistral_client.chat.complete(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": query,
                        },
                    ],
                )


            elif 'gpt' in model_name:
                gpt_query = formatted_k_shot_string + '\n' + inference_prompt
                gpt_query = f"{gpt_query.replace('{}', f'{{{neutral_sentence}}}')}"
                query = gpt_query
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content":query}]
                chat_response = gpt_client.chat.completions.create(
                    model = model_name,
                    messages = messages,
                    temperature = 0.2,
                )


            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            final_output.append({
                'id_neutral_sentence': int(sentence['sentenceid']),
                'neutral_sentence': sentence['sentences'],
                'username': username,
                'tst_id': username + timestamp,
                'llm_tst': chat_response.choices[0].message.content,
                "query": query,
                "model": chat_response.model,
                "prompt_tokens": chat_response.usage.prompt_tokens,
                "completion_tokens": chat_response.usage.completion_tokens,
                "object": chat_response.object,
                "promptID": str(prompt_id),
                "timestamp": timestamp,
                "output_run": output_run
            })
            # final_output list to csv file
            df_output = pd.DataFrame(final_output)
            # Datetime information represented as string as a timestamp
            df_output.to_csv(output_llm_folder_path + "s_" + str(sentence['sentenceid']) + "_u_" + username + "_t_" + timestamp + '.csv', index=False)
            df_output_all = pd.concat([df_output_all, df_output], ignore_index=True)
    
    # #try to execute this method, if it fails return df_mistral_output_all
    # try:
    #     df_postprocess = postprocess_llm_tst(df_output_all, output_run)
    # except:
    #     print('Postprocessing failed, returning the raw data. Please run the postprocessing method manually.')
    #     return df_output_all


    return df_output_all,output_run


def postprocess_llm_tst(df,output_run):
    print('Postprocessing LLM TST data')
    # apply try except block to catch exceptions
    try:
        df['tst_sentence'] = df['llm_tst'].apply(lambda x: extract_tst(x)[0])
        df['explanation'] = df['llm_tst'].apply(lambda x: extract_tst(x)[1])

        df = df[['username','id_neutral_sentence','neutral_sentence','tst_id','tst_sentence','explanation','llm_tst','query','model','prompt_tokens','completion_tokens','object','promptID','timestamp','output_run']]
        
        output_llm_folder_path = 'f6_llm_tst_data/' + output_run + '/'

        df.to_csv(output_llm_folder_path + output_run + '_tst_postprocess.csv', index=False)
    except:
        print('Postprocessing failed, returning the raw data. Please run the postprocessing method manually.')
        return df


    return df


def extract_tst(text):

    tst_sentence = text.split('explanation:')[0].split(': ')[1].replace('"', '').replace('\n', '')
    explanation = text.split('explanation:')[1]

    return tst_sentence, explanation


# Evaluation methods
def llm_evl(df,user_sentences,model_name):
    config = configparser.ConfigParser()
    config.read('config.ini')

    if 'mistral' in model_name:
        api_key_mistral = config.get('credentials', 'api_key_mistral')
        # mistral_client = MistralClient(api_key = api_key_mistral)
        mistral_client = Mistral(api_key = api_key_mistral)
    elif 'gpt' in model_name:
        api_key_gpt = config.get('credentials', 'api_key_openai')
        gpt_client = OpenAI(api_key = api_key_gpt)

    surfdrive_url_evaluation_prompts = config.get('credentials', 'surfdrive_url_evaluation_prompts')
    df_eval_prompts = pd.read_csv(surfdrive_url_evaluation_prompts, sep = ';', on_bad_lines='skip').reset_index()
    
    # saving eval outcomes to temp list, to be appended to the final dataframe
    eval_output = []

    for _, row_sentences in stqdm(df.iterrows(),total=df.shape[0],desc = "Evaluating TST sentences"):
    # for _, row_sentences in df.iterrows():

        # take the sentence from the corpus
        sentence = row_sentences['tst_sentence']

        # evaluate the sentence on all evaluation metrics
        for _, row_eval in df_eval_prompts.iterrows():
            eval_promptID = int(row_eval['eval_promptID'])
            user_s = user_sentences[(user_sentences.userid == row_sentences['username']) & (user_sentences.sentenceid ==  row_sentences['id_neutral_sentence'])]['rewritten'].iloc[0]
            prompt_system = row_eval['prompt_system']
            prompt_main = row_eval['prompt_main']
            prompt_inference = row_eval['prompt_inference']
            # if eval_promptID in 1-4
            if eval_promptID in range(0,5):
                formatted_inference = prompt_inference.format(sentence)
                eval_prompt = f"{prompt_system}{prompt_main}{formatted_inference}"
            else:
                formatted_inference = prompt_inference.format(user_s,sentence)
                eval_prompt = f"{prompt_system}{prompt_main}{formatted_inference}"

            # query = [ChatMessage(role="user", content=eval_prompt)]

            # chat_response = mistral_client.chat(
            #     model = model_name,
            #     messages = query,
            # )
            if 'mistral' in model_name:
                # No streaming
                chat_response = mistral_client.chat.complete(
                    model=model_name,
                    messages=[
                            {
                                "role": "user",
                                "content": eval_prompt,
                            },
                        ],
                )
            elif 'gpt' in model_name:
                print('evaluation with gpt')
                # gpt query
                messages = [{"role": "system", "content": "You are an linguistic expert that should evaluate text on different metrics."},
                             {"role": "user", "content": eval_prompt}]
                chat_response = gpt_client.chat.completions.create(
                    model = model_name,
                    messages = messages,
                    temperature = 0.2,
                )



            eval_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            eval_output.append({
                'username': row_sentences['username'],
                'id_neutral_sentence': row_sentences['id_neutral_sentence'],
                'tst_id': row_sentences['tst_id'],
                'tst_sentence': sentence,
                'user_sentence': user_s,
                'eval_id': row_sentences['username'] + eval_timestamp,
                'llm_eval': chat_response.choices[0].message.content,
                "query": eval_prompt,
                "model": chat_response.model,
                "prompt_tokens": chat_response.usage.prompt_tokens,
                "completion_tokens": chat_response.usage.completion_tokens,
                "object": chat_response.object,
                "eval_promptID":  row_eval['eval_promptID'],
                "eval_timestamp": eval_timestamp,
                "output_run": row_sentences['output_run']
            })
            


    df_eval_output = pd.DataFrame(eval_output)

    try:
        output_run = df_eval_output['output_run'].iloc[0]
    except:
        print('evaluation error')
        output_run = str(df['eval_timestamp'].min())

    output_llm_eval_folder_path = 'f8_llm_evaluation_data/' + output_run + '/'
    

    # create the folder if it does not exist
    if not os.path.exists(output_llm_eval_folder_path):
        os.makedirs(output_llm_eval_folder_path)

    df_eval_output.to_csv(output_llm_eval_folder_path + 'eval_' + output_run + '.csv', index=False)
   
    #try to execute this method, if it fails return df_mistral_output_all
    try:
        df_postprocess = postprocess_llm_evl(df_eval_output,output_run)
    except:
        print('Evaluation postprocessing failed, returning the raw data. Please run the evaluation postprocessing method manually.')
        return df_eval_output
    


    return df_postprocess,df_eval_output


def postprocess_llm_evl(df,output_run):
    # create empty list to store all evaluation data
    eval_output_list = []
    output_llm_eval_folder_path = 'f8_llm_evaluation_data/' + output_run + '/'
    count_exceptions = 0

    grouped_data = df.groupby('tst_id')
    for tst_id, group in grouped_data:
        # for each tst_id, create a new row  in the df_all_eval dataframe
        # first, store the tst_id in the new row    
        eval_output = {
            'tst_id': tst_id,
            'tst_sentence': group['tst_sentence'].iloc[0],
            'username': group['username'].iloc[0],
            'id_neutral_sentence': group['id_neutral_sentence'].iloc[0],
            'user_sentence': group['user_sentence'].iloc[0],
        }

        
        for index, row_eval in group.iterrows():
            # append the eval_pID to the new row
            eval_pID = row_eval['eval_promptID']


            if(eval_pID == 4):
                try:
                    eval_output['eval_score_fluency'] = re.findall(r'\d+', row_eval['llm_eval'].split('xplanation')[0])[0]
                    eval_output['timestamp_score_fluency'] = row_eval['eval_timestamp']
                except:
                    eval_output['eval_score_fluency'] = None
                    eval_output['timestamp_score_fluency'] = None
                    print('Exception at index:', index,' \n with value:', row_eval['llm_eval'])
                    count_exceptions += 1

                try:
                    eval_output['eval_score_comprehensibility'] = re.findall(r'\d+', row_eval['llm_eval'].split('xplanation')[0])[1]
                    eval_output['timestamp_score_comprehensibility'] = row_eval['eval_timestamp']
                except:
                    eval_output['eval_score_comprehensibility'] = None
                    eval_output['timestamp_score_comprehensibility'] = None
                    print('Exception at index:', index,' \n with value:', row_eval['llm_eval'])
                    count_exceptions += 1

                try:
                    eval_output['eval_explanation_fluency_comprehensibility'] = row_eval['llm_eval'].split('xplanation=')[1]
                except:
                    eval_output['eval_explanation_fluency_comprehensibility'] = None
                    print('Exception at index:', index,' \n with value:', row_eval['llm_eval'])
                    count_exceptions += 1

            else:
                eval_label = get_eval_label(eval_pID)
                try:
                    eval_output['eval_score_' + eval_label] = re.findall(r'\d+', row_eval['llm_eval'].split('xplanation')[0])[0]
                    eval_output['timestamp_score_' + eval_label] = row_eval['eval_timestamp']
                except:
                    eval_output['eval_score_' + eval_label] = None
                    eval_output['timestamp_score_' + eval_label] = None
                    print('Exception at index:', index,' \n with value:', row_eval['llm_eval'])
                    count_exceptions += 1
                
                try:
                    eval_output['eval_explanation_' + eval_label] = row_eval['llm_eval'].split('xplanation=')[1]
                    eval_output['timestamp_score_' + eval_label] = row_eval['eval_timestamp']

                except:
                    eval_output['eval_explanation_' + eval_label] = None
                    eval_output['timestamp_score_' + eval_label] = None
                    print('Exception at index:', index,' \n with value:', row_eval['llm_eval'])
                    count_exceptions += 1
            
        eval_output['output_run'] = output_run
        
        # append the new row with the evaluation scores to the eval_output_list
        eval_output_list.append(eval_output)


    df_eval_output = pd.DataFrame(eval_output_list)

    df_eval_output.to_csv(output_llm_eval_folder_path + 'postprocess_eval_' + output_run + '.csv', index=False)

    print('Number of exceptions:', count_exceptions)

    return df_eval_output

def get_eval_label(int_label):
    # return eval string based on 0 to 4 switch logic
    switcher = {
        0: "formality",
        1: "descriptiveness",
        2: "emotionality",
        3: "sentiment",
        5: "topic_similarity",
        6: "meaning_similarity"
    }
    return switcher.get(int_label, "Invalid label")

def extract_text_between_quotes(string):
    pattern = r'"([^"]*)"'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return None
