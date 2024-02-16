import pandas as pd
import glob
from openai import OpenAI
import configparser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json
from wordcloud import WordCloud
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def extract_info(row):
    try:
        # Preprocess the string to replace double backslashes with a single backslash
        cleaned_row = row.replace('\\', '').replace('\\\\', '')
        json_data = json.loads(cleaned_row)
        return pd.Series({
            'rewritten_sentence': json_data.get('rewritten_sentence', ''),
            'style_keywords': json_data.get('style_keywords', []),
            'explanation': json_data.get('explanation', ''),
            'syntax_keywords': json_data.get('syntax_keywords', []),
        })
    except Exception as e:
        return pd.Series({
            'rewritten_sentence': '',
            'style_keywords': [],
            'explanation': '',
            'syntax_keywords': [],
        })



def get_mistral_medium_output(input_sentences,username):
    parallel_folder = 'output_parallel_data/'
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.ini')
    api_key_mistral = config.get('credentials', 'api_key_mistral')
    mistral_m = "mistral-medium"
    mistral_client = MistralClient(api_key = api_key_mistral)
    prompt_id = '111'
    prompt_content = '''
    [INST]You are an expert in text style transfer. 
    Here is a sentence written in the style of person X {}. 
    Your task is four-fold. First, rewrite the same sentence without any style. 
    Second, describe the conversational style of that person using up to 5 keywords. 
    Third, explain the conversational style of person X' sentence.
    Forth, describe the syntax of person X' using up to 5 keywords.
    The output needs to be formated as a valid JSON object with the following fields: 
    rewritten_sentence, style_keywords, explanation, syntax_keywords
    [\INST]'''
    
    final_output = []
    
    for i in range(0,len(input_sentences)):   
        original = input_sentences[i]
        query = f"{prompt_content.replace('{}', f'{{{original}}}',1)}"
        
        messages = [ ChatMessage(role = "user", content = query) ]
        
        # No streaming
        chat_response = mistral_client.chat(
            model = mistral_m,
            messages = messages,
            safe_prompt=True
        )
        
        final_output.append({'original': original,'output': chat_response.choices[0].message.content,"model": chat_response.model, "prompt_tokens" : chat_response.usage.prompt_tokens,"completion_tokens" : chat_response.usage.completion_tokens,"object" : chat_response.object, "promptID" : prompt_id})

    
    df_mistral_output = pd.DataFrame(final_output)
    df_mistral_output = df_mistral_output.reset_index(names='messageID')
    # Apply the function to each row and create new columns
    df_mistral_output[['rewritten_sentence', 'style_keywords', 'explanation', 'syntax_keywords']] = df_mistral_output['output'].apply(extract_info)
    df_mistral_output = df_mistral_output[['messageID','original','rewritten_sentence','style_keywords','explanation', 'syntax_keywords','output']]
    df_mistral_output.to_csv(parallel_folder + username + '_parallel_data_mistral_medium.csv')

    return df_mistral_output


def get_gpt_4_output(input_sentences,username):
    parallel_folder = 'output_parallel_data/'
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.ini')
    api_key_openai = config.get('credentials', 'api_key_openai')
    gtp_client = OpenAI(api_key = api_key_openai)
    gpt_m = "gpt-4"
    gpt_system_msg = '''
    You are an expert in text style transfer. 
    Your task is four-fold. First, rewrite the sentence of a person X without any style. 
    Second, describe the conversational style of that person using up to 5 keywords. 
    Third, explain the conversational style of person X' sentence.
    Forth, describe the syntax of person X' using up to 5 keywords.
    The output needs to be formated as a valid JSON object with the following fields: 
    rewritten_sentence, style_keywords, explanation, syntax_keywords 
    '''
    gpt_temperature = 0.2
    gpt_max_tokens = 256
    gpt_frequency_penalty = 0.0
    prompt_id = '111'
    prompt_content = '''
    Here is a sentence written in the style of person X {}.  
    '''

    final_output = []
    for i in range(0,len(input_sentences)):          
        original = input_sentences[i]
        query = f"{prompt_content.replace('{}', f'{{{original}}}')}"
        
        message=[{"role": "system", "content": gpt_system_msg}, {"role": "user", "content":query}]
        
        chat_response = gtp_client.chat.completions.create(
            model = gpt_m,
            messages = message,
            temperature = gpt_temperature,
            max_tokens = gpt_max_tokens,
            frequency_penalty = gpt_frequency_penalty
        )
        final_output.append({'original': original,'output': chat_response.choices[0].message.content,"model": chat_response.model, "prompt_tokens" : chat_response.usage.prompt_tokens,"completion_tokens" : chat_response.usage.completion_tokens,"object" : chat_response.object, "promptID" : prompt_id, "temperature": gpt_temperature})
        
    df_gpt_output = pd.DataFrame(final_output)
    df_gpt_output = df_gpt_output.reset_index(names='messageID')
    df_gpt_output[['rewritten_sentence', 'style_keywords', 'explanation', 'syntax_keywords']] = df_gpt_output['output'].apply(extract_info)
    df_gpt_output = df_gpt_output[['messageID','original','rewritten_sentence','style_keywords','explanation', 'syntax_keywords','output']]
    df_gpt_output.to_csv(parallel_folder + username + '_parallel_data_gpt_4.csv')
        

def get_keyword_details():
    parallel_folder = 'output_parallel_data/'
    csv_files = glob.glob(parallel_folder + '*')
    
    usernames= []
    median_word_count_list = []
    num_sentences = []
    
    
    for file in csv_files:
        # Check if the file ends with 'chat_llm.csv'
    
        if file.endswith('parallel_data_mistral_medium.csv'):
            username = file[21:23]
            df_test = pd.read_csv(file)
            # print('Username ', username, '\n MISTRAL Keywords:',df_test['syntax_keywords'])
            style_keywords_lists = [ast.literal_eval(keyword_str) for keyword_str in list(df_test['style_keywords'])]
            # Merge all lists into one
            all_style_keywords = [keyword for sublist in style_keywords_lists for keyword in sublist]
            # Create a DataFrame with keyword and frequency columns
            style_keywords_df = pd.DataFrame({'keyword': all_style_keywords})
            # Count the frequency of each keyword
            style_keyword_counts = style_keywords_df['keyword'].value_counts().reset_index()
            # Rename the columns
            style_keyword_counts.columns = ['keyword', 'frequency']
            # Add a 'percentage' column
            total_style_keywords = style_keyword_counts['frequency'].sum()
            style_keyword_counts['percentage'] = (style_keyword_counts['frequency'] / total_style_keywords) * 100
            # Sort the DataFrame by frequency in descending order
            style_keyword_counts = style_keyword_counts.sort_values(by='frequency', ascending=False)
            # Display the final DataFrame
            style_keyword_counts.to_csv(parallel_folder + username + '_mistral_style_keywords_details.csv')
            
            # Create a WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_style_keywords))
            # Plot the WordCloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Mistral Style Keyword Word Cloud. User ' + username)
            # Save the WordCloud image to a PNG file
            plt.savefig(parallel_folder + username + '_mistral_style_keyword_cloud_image.png', bbox_inches='tight')
            matplotlib.pyplot.close()
            
            # NOW THE SYNTAX KEYWORDS
            syntax_keywords_lists = [ast.literal_eval(keyword_str) for keyword_str in list(df_test['syntax_keywords'])]
            # Merge all lists into one
            all_syntax_keywords = [keyword for sublist in syntax_keywords_lists for keyword in sublist]
            # Create a DataFrame with keyword and frequency columns
            syntax_keywords_df = pd.DataFrame({'keyword': all_syntax_keywords})
            # Count the frequency of each keyword
            syntax_keyword_counts = syntax_keywords_df['keyword'].value_counts().reset_index()
            # Rename the columns
            syntax_keyword_counts.columns = ['keyword', 'frequency']
            # Add a 'percentage' column
            total_syntax_keywords = syntax_keyword_counts['frequency'].sum()
            syntax_keyword_counts['percentage'] = (syntax_keyword_counts['frequency'] / total_syntax_keywords) * 100
            # Sort the DataFrame by frequency in descending order
            syntax_keyword_counts = syntax_keyword_counts.sort_values(by='frequency', ascending=False)
            # Display the final DataFrame
            syntax_keyword_counts.to_csv(parallel_folder + username + '_mistral_syntax_keywords_details.csv')
            
            # Create a WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_syntax_keywords))
            # Plot the WordCloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Mistral Syntax Keyword Word Cloud. User ' + username)
            # Save the WordCloud image to a PNG file
            plt.savefig(parallel_folder + username + '_mistral_syntax_keyword_cloud_image.png', bbox_inches='tight')
            matplotlib.pyplot.close()
            
        elif file.endswith('parallel_data_gpt_4.csv'):
            username = file[21:23]
            df_test = pd.read_csv(file)
            # print('Username ', username, '\n GPT Keywords:',df_test['syntax_keywords'])
            style_keywords_lists = [list(map(str.lower, ast.literal_eval(keyword_str))) for keyword_str in list(df_test['style_keywords'])]
            # Merge all lists into one
            all_style_keywords = [keyword for sublist in style_keywords_lists for keyword in sublist]
            # Create a DataFrame with keyword and frequency columns
            style_keywords_df = pd.DataFrame({'keyword': all_style_keywords})
            # Count the frequency of each keyword
            style_keyword_counts = style_keywords_df['keyword'].value_counts().reset_index()
            # Rename the columns
            style_keyword_counts.columns = ['keyword', 'frequency']
            # Add a 'percentage' column
            total_style_keywords = style_keyword_counts['frequency'].sum()
            style_keyword_counts['percentage'] = (style_keyword_counts['frequency'] / total_style_keywords) * 100
            # Sort the DataFrame by frequency in descending order
            style_keyword_counts = style_keyword_counts.sort_values(by='frequency', ascending=False)
            # Display the final DataFrame
            style_keyword_counts.to_csv(parallel_folder + username + '_gpt_4_style_keywords_details.csv')
            
            # Create a WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_style_keywords))
            # Plot the WordCloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('GPT4 Style Keyword Word Cloud. User ' + username)
            # Save the WordCloud image to a PNG file
            plt.savefig(parallel_folder + username + '_gpt_4_style_keyword_cloud_image.png', bbox_inches='tight')
            matplotlib.pyplot.close()
            
            syntax_keywords_lists = [list(map(str.lower, ast.literal_eval(keyword_str))) for keyword_str in list(df_test['syntax_keywords'])]
            # Merge all lists into one
            all_syntax_keywords = [keyword for sublist in syntax_keywords_lists for keyword in sublist]
            # Create a DataFrame with keyword and frequency columns
            syntax_keywords_df = pd.DataFrame({'keyword': all_syntax_keywords})
            # Count the frequency of each keyword
            syntax_keyword_counts = syntax_keywords_df['keyword'].value_counts().reset_index()
            # Rename the columns
            syntax_keyword_counts.columns = ['keyword', 'frequency']
            # Add a 'percentage' column
            total_syntax_keywords = syntax_keyword_counts['frequency'].sum()
            syntax_keyword_counts['percentage'] = (syntax_keyword_counts['frequency'] / total_syntax_keywords) * 100
            # Sort the DataFrame by frequency in descending order
            syntax_keyword_counts = syntax_keyword_counts.sort_values(by='frequency', ascending=False)
            # Display the final DataFrame
            syntax_keyword_counts.to_csv(parallel_folder + username + '_gpt_4_syntax_keywords_details.csv')

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_syntax_keywords))
            # Plot the WordCloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('GPT4 Syntax Keyword Word Cloud. User ' + username)
            # Save the WordCloud image to a PNG file
            plt.savefig(parallel_folder + username + '_gpt_4_syntax_keyword_cloud_image.png', bbox_inches='tight')
            matplotlib.pyplot.close()
            
def delete_after_character(input_string, character):
    index = input_string.find(character)
    if index != -1:  # Check if the character is found in the string
        return input_string[:index + 1]  # Include the character itself
    else:
        return input_string  # Return the original string if the character is not found
        
def postprocess_rows(row):
    try:
        row = delete_after_character(row,'}')
        # Preprocess the string to replace double backslashes with a single backslash
        cleaned_row = row.replace('\\', '').replace('\\\\', '')
        json_data = json.loads(cleaned_row)
        return pd.Series({
            'rewritten_sentence': json_data.get('rewritten_sentence', ''),
            'style_keywords': json_data.get('style_keywords', []),
            'explanation': json_data.get('explanation', ''),
            'syntax_keywords': json_data.get('syntax_keywords', []),
        })
    except Exception as e:
        return pd.Series({
            'rewritten_sentence': '',
            'style_keywords': [],
            'explanation': '',
            'syntax_keywords': [],
        })

def postprocess_files():
    output_parallel_data = 'output_parallel_data/'
    csv_files = glob.glob(output_parallel_data + '/*.csv')

    for file in csv_files:
        # Check if the file ends with 'chat_llm.csv'
        if file.endswith('mistral_medium.csv'):       
            columns_to_update = ['rewritten_sentence', 'style_keywords','explanation','syntax_keywords']
            df_file = pd.read_csv(file)
            df_file_temp = df_file[df_file['rewritten_sentence'].isnull()]
            # there are no incomplete rows..
            if df_file_temp.empty:
                return
            else:
                df_file_temp[columns_to_update] = df_file_temp['output'].apply(postprocess_rows)
                # Specify the columns you want to update 
                # Merge both dfs
                df_file_merged = pd.merge(df_file, df_file_temp, on='messageID', how='left', suffixes=('', '_new'))
                # Update the specified columns in df_file
                for col in columns_to_update:
                    df_file[col] = df_file_merged[col+'_new'].fillna(df_file_merged[col])
                
                df_file.to_csv(file)
    
    
    

