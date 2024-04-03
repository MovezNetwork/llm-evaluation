import pandas as pd
import glob
from openai import OpenAI
import configparser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json
import random
import os
os.environ["WANDB_SILENT"] = "true"
from wandb.sdk.data_types.trace_tree import Trace
import wandb
import re

def generate_x_shots_files():
    output_parallel_data = 'f2_prompt_ready_chat_data/'
    output_shots_data = 'f4_shots_data/'
    seed_value = 42  
    # read the messages input files
    csv_files = glob.glob(output_parallel_data + '*')
    random.seed(seed_value)
    
    for file in csv_files:
        #read the username
        username = file.split('/')[1][:2]
        if file.endswith('mistral_medium.csv'):
            df_mistral = pd.read_csv(file)
            possible_rows = list(range(df_mistral.shape[0]))
            
            random.shuffle(possible_rows)
            three_shots = possible_rows[:3]
            
            random.shuffle(possible_rows)
            five_shots = possible_rows[:5]

            random.shuffle(possible_rows)
            ten_shots = possible_rows[:10]
             
            lst_x_shots_mistral = [df_mistral.iloc[three_shots],df_mistral.iloc[five_shots],df_mistral.iloc[ten_shots]]
            
            for df_shots in lst_x_shots_mistral:
                 #save the shots dataset to csv
                df_shots[['messageID','rewritten_sentence','original']].to_csv(output_shots_data  + username + '_mistral_shots_' + str(df_shots.shape[0])+'.csv',index=False)

            #now get the appropriate gpt file
            gpt_file = output_parallel_data + username + '_parallel_data_gpt_4.csv'
            df_gpt = pd.read_csv(gpt_file)
            lst_x_shots_gpt = [df_gpt.iloc[three_shots],df_gpt.iloc[five_shots],df_gpt.iloc[ten_shots]]
            for df_shots in lst_x_shots_gpt:
                 #save the shots dataset to csv
                df_shots[['messageID','rewritten_sentence','original']].to_csv(output_shots_data  + username + '_gpt_shots_' + str(df_shots.shape[0])+'.csv',index=False)
            
def extract_info(row):
    try:
        # Preprocess the string to replace double backslashes with a single backslash
        cleaned_row = row.replace('\\', '').replace('\\\\', '')
        # Check if cleaned_row is a valid JSON
        json.loads(cleaned_row)  # This line is just to check if it raises an exception
        json_data = json.loads(cleaned_row)
        return pd.Series({
            'rewritten_sentence': json_data.get('rewritten_sentence', ''),
            'explanation': json_data.get('explanation', '')
        })
    except Exception as e:
        # If it's not a valid JSON, return empty rows
        return pd.Series({
            'rewritten_sentence': [],
            'explanation': []
        })
        
def prompting_mistral(prompt_id,x_shots,mistral_m,input_sentences,save_online,parallel_data,context):
    config = configparser.ConfigParser()
    # Read the configuration file & paths
    config.read('config.ini')
    api_key_mistral = config.get('credentials', 'api_key_mistral')
    surfdrive_url_prompts = config.get('credentials', 'surfdrive_url_prompts')
    output_llm_folder_path = 'f6_llm_tst_data/'
    output_shots_data = 'f4_shots_data/'
    output_context = 'f3_context_data/'    
    
    # reading prompt template components - depends on prompt_id
    df_prompts = pd.read_csv(surfdrive_url_prompts,sep=';').reset_index()
    
    mistral_prompt_system_content = df_prompts['prompt_system_content'].iloc[prompt_id]
    mistral_prompt_x_shot_template = df_prompts['prompt_x_shot_template'].iloc[prompt_id]
    mistral_prompt_content_addition = df_prompts['prompt_content_addition'].iloc[prompt_id]
    prompt_id = str(df_prompts['promptID'].iloc[prompt_id])

    dict_mistral_output = {}
    csv_files = glob.glob(output_shots_data + '*')
    for file in csv_files:
        if (file.find('mistral') != -1 and file.endswith(str(x_shots)+'.csv')):
            # run setup
            username = file[20:22]
            run_id = str(random.randint(100000, 999999))            
            df_shots = pd.read_csv(file)
            print("user_" + username + "_promptID_" + prompt_id + '_model_'+ mistral_m + '_shots_' + str(x_shots) + "_run_id_" + run_id)

            # Update the prompt template with the x-shot sentences
            x_shots_list = []
            messages_id = []

            # The prompt uses parallel data prompt
            if parallel_data:           
                for index, row in df_shots.iterrows():
                    # Access values in the desired order and append to the list
                    x_shots_list.append(row['rewritten_sentence'])
                    x_shots_list.append(row['original'])  
                    messages_id.append(row['messageID'])        
                formatted_string = mistral_prompt_system_content + '\n'
                for i in range(0, len(x_shots_list), 2):
                    formatted_string += mistral_prompt_x_shot_template.format(x_shots_list[i], x_shots_list[i + 1]) + "\n\n"
            # Non-parallel data prompt prompt
            else:
                for index, row in df_shots.iterrows():
                    # Access values in the desired order and append to the list
                    x_shots_list.append(row['original'])  
                    messages_id.append(row['messageID'])        
                formatted_string = mistral_prompt_system_content + '\n'
                for i in range(0, len(x_shots_list)):
                    formatted_string += mistral_prompt_x_shot_template.format(x_shots_list[i]) + "\n\n"

            context_string = ''
            # context is optional
            if context:
                with open(output_context + username + '_mistral_context.txt', "r") as f:
                    context_string = f.read()
            
            
            formatted_string += '\n' + context_string + '\n'
            formatted_string += mistral_prompt_content_addition
            
            
            # Query Mistral API
            mistral_client = MistralClient(api_key = api_key_mistral)
            final_output = []
            for i in range(0,len(input_sentences)):
                query = f"{formatted_string.replace('{}', f'{{{input_sentences[i]}}}')}"
                messages = [ ChatMessage(role = "user", content = query) ]
                # No streaming
                chat_response = mistral_client.chat(
                    model = mistral_m,
                    messages = messages,
                )
                final_output.append({'original': input_sentences[i],'rewritten_sentence': extract_info(chat_response.choices[0].message.content)['rewritten_sentence'],'explanation' : extract_info(chat_response.choices[0].message.content)['explanation'], 'output': chat_response.choices[0].message.content,"query":query, "model": chat_response.model, "prompt_tokens" : chat_response.usage.prompt_tokens,"completion_tokens" : chat_response.usage.completion_tokens,"object" : chat_response.object, "promptID" : prompt_id})

            # Save mistral output in a csv (locally), and Weights&Biases (online, optional)
            df_mistral_output = pd.DataFrame(final_output)
            
            # # EXTRA SENTENCE LOGIC
            # file_finder = "user_" + username + "_promptID_" + prompt_id + '_model_'+ mistral_m + '_shots_' + str(df_shots.shape[0])
            # files_affected = 0
            # for filename in os.listdir(output_llm_folder_path):
            #     if filename.endswith(".csv") and file_finder in filename:
            #         files_affected = files_affected + 1
            #         # File matches the pattern, open it as a DataFrame
            #         file_path = os.path.join(output_llm_folder_path, filename)
            #         df = pd.read_csv(file_path)
            #         updated_df = pd.concat([df, df_mistral_output], ignore_index=True)
            #         updated_df.to_csv(output_llm_folder_path+filename, index=False)
            #         print('Updated: ',filename)
            # print('Files affected: ',files_affected)
            
            df_mistral_output.to_csv(output_llm_folder_path + "user_" + username + "_promptID_" + prompt_id + '_model_'+ mistral_m + '_shots_' + str(df_shots.shape[0]) + "_run_id_" + run_id + '_output.csv', index=False)

            dict_mistral_output[username] = df_mistral_output


            if save_online:
                wandb.init(project="lmm-evaluate", name="user_" + username + "_promptID_" + prompt_id + '_model_'+ mistral_m+ '_shots_' + str(x_shots) + "_run_id_" + run_id , mode = "disabled")
                # log df as a table to W&B for interactive exploration
                wandb.log({"promptID_" + prompt_id + '_model'+ mistral_m + "_run_id_" + run_id : wandb.Table(dataframe = df_mistral_output)})
                # log csv file as an dataset artifact to W&B for later use
                artifact = wandb.Artifact('df_' +"run_id_" + run_id + "promptID_" + prompt_id + '_model_'+ mistral_m + '_shots_' + str(x_shots) + '_output', type="dataset")
                artifact.add_file(output_llm_folder_path + "user_" + username + "_promptID_" + prompt_id + '_model_'+ mistral_m + '_shots_' + str(x_shots) + "_run_id_" + run_id  + '_output.csv')
                wandb.log_artifact(artifact)
                wandb.finish()

    return dict_mistral_output

def prompting_gpt(prompt_id,x_shots,gpt_m,input_sentences,save_online,parallel_data,context):
    config = configparser.ConfigParser()
    # Read the configuration file & paths
    config.read('config.ini')
    api_key_gpt = config.get('credentials', 'api_key_openai')
    surfdrive_url_prompts = config.get('credentials', 'surfdrive_url_prompts')
    output_llm_folder_path = 'f6_llm_tst_data/'
    output_shots_data = 'f4_shots_data/'
    output_context = 'f3_context_data/'
    gpt_temperature = 0.2
    
    # reading prompt template components - depends on prompt_id
    df_prompts = pd.read_csv(surfdrive_url_prompts,sep=';').reset_index()
    
    gpt_prompt_system_content = df_prompts['prompt_system_content'].iloc[prompt_id]
    gpt_prompt_x_shot_template = df_prompts['prompt_x_shot_template'].iloc[prompt_id]
    gpt_prompt_content_addition = df_prompts['prompt_content_addition'].iloc[prompt_id]
    prompt_id = str(df_prompts['promptID'].iloc[prompt_id])
    
    dict_gpt_output = {}
    
    csv_files = glob.glob(output_shots_data + '*')
    for file in csv_files:
        if (file.find('gpt') != -1 and file.endswith(str(x_shots)+'.csv')):
            # run setup
            username = file[20:22]
            run_id = str(random.randint(100000, 999999))            
            df_shots = pd.read_csv(file)

            gpt_final_path = str(output_llm_folder_path) + "user_" + str(username) + "_promptID_" + str(prompt_id) + '_model_'+ str(gpt_m) + '_shots_' + str(df_shots.shape[0]) + "_run_id_" + str(run_id) + '_output.csv'
            print(gpt_final_path)
            # Update the prompt template with the x-shot sentences
            x_shots_list = []
            messages_id = []

            # The prompt uses parallel data
            if parallel_data:
                for index, row in df_shots.iterrows():
                    # Access values in the desired order and append to the list
                    x_shots_list.append(row['rewritten_sentence'])
                    x_shots_list.append(row['original'])  
                    messages_id.append(row['messageID'])        
                formatted_string = ''
                for i in range(0, len(x_shots_list), 2):
                    formatted_string += gpt_prompt_x_shot_template.format(x_shots_list[i], x_shots_list[i + 1]) + "\n\n"
                
            # Non-parallel data prompt prompt
            else:
                for index, row in df_shots.iterrows():
                    # Access values in the desired order and append to the list
                    x_shots_list.append(row['original'])  
                    messages_id.append(row['messageID'])        
                formatted_string = ''
                for i in range(0, len(x_shots_list)):
                    formatted_string += gpt_prompt_x_shot_template.format(x_shots_list[i]) + "\n\n"
                
            
            context_string = ''
            # context is optional
            if context:
                with open(output_context + username + '_gpt_context.txt', "r") as f:
                    context_string = f.read()
                
            formatted_string += '\n' + context_string + '\n'
            
            formatted_string += gpt_prompt_content_addition
            # Query gpt API
            gpt_client = OpenAI(api_key = api_key_gpt)
            final_output = []
            for i in range(0,len(input_sentences)):
                query = f"{formatted_string.replace('{}', f'{{{input_sentences[i]}}}')}"
                message = [{"role": "system", "content": gpt_prompt_system_content}, {"role": "user", "content":query}]
                # No streaming
                chat_response = gpt_client.chat.completions.create(
                    model = gpt_m,
                    messages = message,
                    temperature = gpt_temperature,
                )
                final_output.append({'original': input_sentences[i],
                                     'rewritten_sentence': extract_info(chat_response.choices[0].message.content)['rewritten_sentence'],
                                     'explanation' : extract_info(chat_response.choices[0].message.content)['explanation'],
                                     'output': chat_response.choices[0].message.content,
                                     "query":query,
                                     "model": chat_response.model,
                                     "prompt_tokens" : chat_response.usage.prompt_tokens,
                                     "completion_tokens" : chat_response.usage.completion_tokens,
                                     "object" : chat_response.object,
                                     "promptID" : prompt_id,
                                     "temperature": gpt_temperature})

            # Save gpt output in a csv (locally), and Weights&Biases (online, optional)
            df_gpt_output = pd.DataFrame(final_output)

            # # EXTRA SENTENCE LOGIC
            # file_finder = "user_" + username + "_promptID_" + prompt_id + '_model_'+ gpt_m + '_shots_' + str(df_shots.shape[0])
            # files_affected = 0
            # for filename in os.listdir(output_llm_folder_path):
            #     if filename.endswith(".csv") and file_finder in filename:
            #         files_affected = files_affected + 1
            #         # File matches the pattern, open it as a DataFrame
            #         file_path = os.path.join(output_llm_folder_path, filename)
            #         df = pd.read_csv(file_path)
            #         updated_df = pd.concat([df, df_gpt_output], ignore_index=True)
            #         updated_df.to_csv(output_llm_folder_path+filename, index=False)
            #         print('Updated: ',filename)
            # print('Files affected: ',files_affected)
            
            df_gpt_output.to_csv(gpt_final_path, index=False)

            dict_gpt_output[username] = df_gpt_output


            if save_online:
                wandb.init(project="lmm-evaluate", name= "user_" + username + "_promptID_" + prompt_id + '_model_'+ gpt_m+ '_shots_' + str(x_shots)+"_run_id_" + run_id , mode = "disabled")
                # log df as a table to W&B for interactive exploration
                wandb.log({"promptID_" + prompt_id + '_model'+ gpt_m + "_run_id_" + run_id : wandb.Table(dataframe = df_gpt_output)})
                # log csv file as an dataset artifact to W&B for later use
                artifact = wandb.Artifact('df_' +"run_id_" + run_id + "promptID_" + prompt_id + '_model_'+ gpt_m + '_shots_' + str(x_shots) + '_output', type="dataset")
                artifact.add_file(output_llm_folder_path  + "user_" + username + "_promptID_" + prompt_id + '_model_'+ gpt_m + '_shots_' + str(x_shots) + "_run_id_" + run_id + '_output.csv')
                wandb.log_artifact(artifact)
                wandb.finish()

    return dict_gpt_output


