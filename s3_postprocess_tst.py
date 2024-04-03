import pandas as pd
import glob
import configparser
import json
import random
import os
import re

def postprocess_rows_gpt_4(row):
    try:
        cleaned_row = row.replace('\\n', ' ')
        fixed_json = '{' + cleaned_row
        split = fixed_json.split('\n')
        if(len(split) == 2):
            fixed_json = split[0] + split[1]
        json_data = json.loads(fixed_json)
        processed = pd.Series({
            'rewritten_sentence': json_data.get('rewritten_sentence', ''),
            'explanation': json_data.get('explanation', '')
        })
        return processed
    except Exception as e:
        print('Exception postprocess_rows_gpt:', e)
        return pd.Series({
            'rewritten_sentence': '',
            'explanation': ''
        })



def postprocess_rows_gpt_3_5(row):
    try:
        cleaned_row = row.replace('\"', ' ')
        fixed_json = '{' + cleaned_row
        split = fixed_json.split('\n')
        if(len(split) == 2):
            fixed_json = split[0] + split[1]
        json_data = json.loads(fixed_json)
        processed = pd.Series({
            'rewritten_sentence': json_data.get('rewritten_sentence', ''),
            'explanation': json_data.get('explanation', '')
        })
        return processed
    except Exception as e:
        print('Exception postprocess_rows_gpt:', e)
        return pd.Series({
            'rewritten_sentence': '',
            'explanation': ''
        })

def postprocess_rows_gpt_3_5_second(row):
    try:
        cleaned_row = row.replace('\\"', '')
        json_data = json.loads(cleaned_row)
        processed = pd.Series({
            'rewritten_sentence': json_data.get('rewritten\_sentence', ''),
            'explanation': json_data.get('explanation', '')
        })
        return processed
    except Exception as e:
        print('Exception postprocess_rows_gpt:', e)
        return pd.Series({
            'rewritten_sentence': '',
            'explanation': ''
        })

def postprocess_rows_mistral(row):
    try:
        cleaned_row = row
        # print('Input row: ', row, '\n')
        if '\\n' in cleaned_row:
            cleaned_row = row.replace('\\n', '')
            
        if '\\' in cleaned_row:
             cleaned_row = cleaned_row.replace('\\', '')

        # this is a json-like string
        if '{' in cleaned_row:    
            # print('After: ', cleaned_row, '\n')
            json_data = json.loads(cleaned_row)
            processed = pd.Series({
                'rewritten_sentence': json_data.get('rewritten_sentence', ''),
                'explanation': json_data.get('explanation', '')
            })
        #some outputs are not at all json-like
        else:
            pattern = r'(?i)explanation:'
            # Find the match using regular expressions
            match = re.search(pattern, cleaned_row)
            
            if match:
                cleaned_row = cleaned_row[:match.start()].strip()
                cleaned_row = cleaned_row.replace('"', '')
                # print('Cleaned: ', cleaned_row, '\n')
            else:
                cleaned_row = row.replace('"', '')
                # print('Cleaned: ', cleaned_row, '\n')
                
            processed = pd.Series({
                'rewritten_sentence': cleaned_row,
                'explanation': ''
            })        
        return processed
    except Exception as e:
        print('Exception postprocess_rows_gpt:', e)
        return pd.Series({
            'rewritten_sentence': '',
            'explanation': ''
        })


def process_json(folder):
    
    csv_files = glob.glob(folder + '*')
    
    columns_to_save_gpt = ['original','rewritten_sentence','explanation','output','query','model','prompt_tokens','completion_tokens','object','promptID','temperature']

    columns_to_save_mistral = ['original','rewritten_sentence','explanation','output','query','model','prompt_tokens','completion_tokens','object','promptID']
    print(len(csv_files))
    for file in csv_files:
        # Its a gpt-4 file
        if file.find('gpt-4')!=-1: 
            columns_to_update = ['rewritten_sentence','explanation']
            df_file = pd.read_csv(file)
            df_file = df_file.reset_index()
            
            # Filter rows where 'column1' and 'column2' strings are the same - this means that the output has just been rewritten to the
            df_file_temp = df_file[df_file['rewritten_sentence'] == df_file['output']]
            # All rows are fine!
            if df_file_temp.empty:
                continue
            else:
                # print('File-fix GPT4 ',file)
                df_file_temp[columns_to_update] = df_file_temp['output'].apply(postprocess_rows_gpt_4)

                df_file_merged = pd.merge(df_file, df_file_temp, on='index', how='left', suffixes=('', '_new'))
                # Update the specified columns in df_file
                for col in columns_to_update:
                    df_file[col] = df_file_merged[col+'_new'].fillna(df_file_merged[col])
                
                df_file[columns_to_save_gpt].to_csv(file)

        elif file.find('gpt-3.5')!=-1: 
            columns_to_update = ['rewritten_sentence','explanation']
            df_file = pd.read_csv(file)
            df_file = df_file.reset_index()

            df_file_temp = df_file[df_file['rewritten_sentence'].isnull()]
            if not df_file_temp.empty:
                 df_file_temp[columns_to_update] = df_file_temp['output'].apply(postprocess_rows_gpt_3_5_second)
                 df_file_merged = pd.merge(df_file, df_file_temp, on='index', how='left', suffixes=('', '_new'))
                # Update the specified columns in df_file
                 for col in columns_to_update:
                     df_file[col] = df_file_merged[col+'_new'].fillna(df_file_merged[col])
                 # print('saving file', file)
                 df_file[columns_to_save_gpt].to_csv(file)               
            # Filter rows where 'column1' and 'column2' strings are the same - this means that the output has just been rewritten to the
            df_file_temp = df_file[(df_file['rewritten_sentence'] == df_file['output'])]
            # All rows are fine!
            if df_file_temp.empty:
                continue
            else:
                # print('File-fix GPT3 ',file)
                df_file_temp[columns_to_update] = df_file_temp['output'].apply(postprocess_rows_gpt_3_5)

                df_file_merged = pd.merge(df_file, df_file_temp, on='index', how='left', suffixes=('', '_new'))
                # Update the specified columns in df_file
                for col in columns_to_update:
                    df_file[col] = df_file_merged[col+'_new'].fillna(df_file_merged[col])
                
                df_file[columns_to_save_gpt].to_csv(file)

        elif (file.find('mistral-small')!=-1 or file.find('mistral-medium')!=-1):
            columns_to_update = ['rewritten_sentence','explanation']
            df_file = pd.read_csv(file)
            df_file = df_file.reset_index()
            
            df_file_temp = df_file[df_file['rewritten_sentence'].isnull()]
            if not df_file_temp.empty:
                print('Null fields Mistral: ',file,'\n')
                df_file_temp[columns_to_update] = df_file_temp['output'].apply(postprocess_rows_mistral)

                df_file_merged = pd.merge(df_file, df_file_temp, on='index', how='left', suffixes=('', '_new'))
                # Update the specified columns in df_file
                for col in columns_to_update:
                    df_file[col] = df_file_merged[col+'_new'].fillna(df_file_merged[col])
                
                df_file[columns_to_save_mistral].to_csv(file)

                
                 
            df_file_temp = df_file[df_file['rewritten_sentence'] == df_file['output']]
            # All rows are fine!
            if df_file_temp.empty:
                continue
            else:
                print('File-fix Mistral: ',file,'\n')
                df_file_temp[columns_to_update] = df_file_temp['output'].apply(postprocess_rows_gpt_3_5_second)

                df_file_merged = pd.merge(df_file, df_file_temp, on='index', how='left', suffixes=('', '_new'))
                # Update the specified columns in df_file
                for col in columns_to_update:
                    df_file[col] = df_file_merged[col+'_new'].fillna(df_file_merged[col])
                
                df_file[columns_to_save_mistral].to_csv(file)


def create_tst_evaluation_ready_files():
    output_llm_folder_path = ('f6_llm_tst_data/')
    output_evaluation_folder_path = 'f7_processed_llm_tst_data/'
    
    prompting_strategies = ['Prompts_NoContext', 'Prompts_Context']
    
    
    for prompting in prompting_strategies:
        output_df = pd.DataFrame()
        
        for subdir, dirs, files in os.walk(output_llm_folder_path+prompting):
            file_id = 0
            for file in files:
                filepath = subdir + os.sep + file
                # filter out relevant info
                if filepath.endswith("output.csv"):
                    # print(file_id)
                    # print(filepath)
                    temp_details = file.split('_')
                    temp_output = pd.read_csv(filepath)[['original', 'rewritten_sentence']]
                    temp_output['fileID'] = file_id
                    temp_output['user'] = temp_details[1]
                    temp_output['promptID'] = temp_details[3] if prompting == 'Prompts_NoContext' else temp_details[4]
                    temp_output['model'] = temp_details[5] if prompting == 'Prompts_NoContext' else temp_details[6]
                    temp_output['shots'] = temp_details[7] if prompting == 'Prompts_NoContext' else temp_details[8]
                    temp_output['runID'] = temp_details[10] if prompting == 'Prompts_NoContext' else temp_details[11]
                    
                    # concat temp dataframe
                    output_df = pd.concat([output_df, temp_output], ignore_index=True)
                    file_id = file_id + 1
                
            # reorder columns
            output_df = output_df[['fileID', 'user', 'promptID', 'model', 'shots', 'runID', 'original', 'rewritten_sentence']]
        
            # save to csv
            output_df.to_csv(output_evaluation_folder_path + prompting + '_complete_output.csv', index=False)
