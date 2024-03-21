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
    parallel_folder = 'f2_parallel_data/'
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
    parallel_folder = 'f2_prompt_ready_chat_data/'
    context_folder = 'f3_context_data/'
    csv_files = glob.glob(parallel_folder + '*')
    
    usernames= []
    median_word_count_list = []
    num_sentences = []
    
    
    for file in csv_files:
        # Check if the file ends with 'chat_llm.csv'
    
        if file.endswith('parallel_data_mistral_medium.csv'):
            username = file.split('/')[1][:2]
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
            style_keyword_counts.to_csv(context_folder + username + '_mistral_style_keywords_details.csv')
            
            # Create a WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_style_keywords))
            # Plot the WordCloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Mistral Style Keyword Word Cloud. User ' + username)
            # Save the WordCloud image to a PNG file
            plt.savefig(context_folder + username + '_mistral_style_keyword_cloud_image.png', bbox_inches='tight')
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
            syntax_keyword_counts.to_csv(context_folder + username + '_mistral_syntax_keywords_details.csv')
            
            # Create a WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_syntax_keywords))
            # Plot the WordCloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Mistral Syntax Keyword Word Cloud. User ' + username)
            # Save the WordCloud image to a PNG file
            plt.savefig(context_folder + username + '_mistral_syntax_keyword_cloud_image.png', bbox_inches='tight')
            matplotlib.pyplot.close()
            
        elif file.endswith('parallel_data_gpt_4.csv'):
            username = file.split('/')[1][:2]
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
            style_keyword_counts.to_csv(context_folder + username + '_gpt_4_style_keywords_details.csv')
            
            # Create a WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_style_keywords))
            # Plot the WordCloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('GPT4 Style Keyword Word Cloud. User ' + username)
            # Save the WordCloud image to a PNG file
            plt.savefig(context_folder + username + '_gpt_4_style_keyword_cloud_image.png', bbox_inches='tight')
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
            syntax_keyword_counts.to_csv(context_folder + username + '_gpt_4_syntax_keywords_details.csv')

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_syntax_keywords))
            # Plot the WordCloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('GPT4 Syntax Keyword Word Cloud. User ' + username)
            # Save the WordCloud image to a PNG file
            plt.savefig(context_folder + username + '_gpt_4_syntax_keyword_cloud_image.png', bbox_inches='tight')
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
    output_parallel_data = 'f2_prompt_ready_chat_data/'
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


def generate_context():
    context_dict = {
        "U0": [],
        "U1": [],
        "U2": [],
        "U3": [],
        "U4": [],
        "U5": [],
        "U6": [],
        "U7": [],
        "U8": [],
        "U9": []
    }
    
    context_dict["U0"] = {}
    context_dict["U1"] = {}
    context_dict["U2"] = {}
    context_dict["U3"] = {}
    context_dict["U4"] = {}
    context_dict["U5"] = {}
    context_dict["U6"] = {}
    context_dict["U7"] = {}
    context_dict["U8"] = {}
    context_dict["U9"] = {}
    
    output_chat_data = "f1_processed_user_chat_data/"
    output_context_folder = 'f3_context_data/'
    
    csv_files = glob.glob(output_chat_data + '*')
    for file in csv_files:
        username = file.split('/')[1][:2]
        
        if(file.find('word_distribution') != -1):
            df_words = pd.read_csv(file) 
            tuples_list = [(length, freq) for length, freq in zip(df_words['word_length'], df_words['percentage'])]
            word_count_distribution = str(tuples_list)
            context_dict[username]['word_distribution'] = word_count_distribution
            # print('Word frequency: ', word_count_distribution, '\n\n')
    
        elif(file.find('chat_llm') != -1):
            df_chat = pd.read_csv(file)
            len_chat = df_chat.shape[0]
            
            # GET ADJECTIVES LIST
            adjectives_lists = [ast.literal_eval(keyword_str) for keyword_str in list(df_chat['adjectives'])]
            # Merge all lists into one
            all_adjectives = [keyword for sublist in adjectives_lists for keyword in sublist]
            adjectives_df = pd.DataFrame({'adjective': all_adjectives})
            # Count the frequency of each keyword
            adjectives_df = adjectives_df['adjective'].value_counts().reset_index()
            # Rename the columns
            adjectives_df.columns = ['adjective', 'frequency']
            total_keywords = adjectives_df['frequency'].sum()
            adjectives_df['percentage'] = round((adjectives_df['frequency'] / len_chat) * 100,2)
            tuples_list = [(length, freq) for length, freq in zip(adjectives_df['adjective'], adjectives_df['percentage'])]
            adjective_distribution = str(tuples_list)
            context_dict[username]['adjectives'] = adjective_distribution
            # print('Adjectives: ',adjective_distribution, '\n\n')
    
            
            # GET FUNCTIONAL WORDS
            functional_words_lists = [ast.literal_eval(keyword_str) for keyword_str in list(df_chat['fuctional_words'])]
            # Merge all lists into one
            all_functional_words = [keyword for sublist in functional_words_lists for keyword in sublist]
            functional_words_df = pd.DataFrame({'functional_words': all_functional_words})
            # Count the frequency of each keyword
            functional_words_df = functional_words_df['functional_words'].value_counts().reset_index()
            # Rename the columns
            functional_words_df.columns = ['functional_words', 'frequency']
            total_keywords = functional_words_df['frequency'].sum()
            functional_words_df['percentage'] = round((functional_words_df['frequency'] / len_chat) * 100,2)
            tuples_list = [(length, freq) for length, freq in zip(functional_words_df['functional_words'], functional_words_df['percentage'])]
            functional_words_distribution = str(tuples_list)
            context_dict[username]['functional_words'] = functional_words_distribution
            # print('Functional Words: ',functional_words_distribution, '\n\n')
    
            # GET PUNCTUATION
            punctuation_lists = [ast.literal_eval(keyword_str) for keyword_str in list(df_chat['punctuations'])]
            # Merge all lists into one
            all_punctuation = [keyword for sublist in punctuation_lists for keyword in sublist]
            punctuation_df = pd.DataFrame({'punctuation': all_punctuation})
            # Count the frequency of each keyword
            punctuation_df = punctuation_df['punctuation'].value_counts().reset_index()
            # Rename the columns
            punctuation_df.columns = ['punctuation', 'frequency']
            total_keywords = punctuation_df['frequency'].sum()
            punctuation_df['percentage'] = round((punctuation_df['frequency'] / len_chat) * 100,2)
            tuples_list = [(length, freq) for length, freq in zip(punctuation_df['punctuation'], punctuation_df['percentage'])]
            punctuation_distribution = str(tuples_list)
            context_dict[username]['punctuations'] = punctuation_distribution
            # print('Punctuation: ',punctuation_distribution, '\n\n')
    
            # GET EMOJIS
            all_emojis = list(df_chat['emojis'])
            all_emojis = [x for x in all_emojis if x == x]
            separated_emojis = []
            for emoji in all_emojis:
                # Check if the emoji is a combination of emojis or an emoticon
                if not all(ord(char) < 128 for char in emoji):
                    # If it's a combination of emojis, split them
                    separated_emojis.extend(list(emoji))
                else:
                    # If it's an emoticon, keep it as is
                    separated_emojis.append(emoji)
           
            emojis_df = pd.DataFrame({'emojis': separated_emojis})
            # Count the frequency of each keyword
            emojis_df = emojis_df['emojis'].value_counts().reset_index()
            # Rename the columns
            emojis_df.columns = ['emojis', 'frequency']
            total_keywords = emojis_df['frequency'].sum()
            # print('Rows',df_chat.shape[0])
            # print('emojis_df ', emojis_df)
            emojis_df['percentage'] = round((emojis_df['frequency'] / len_chat) * 100,2)
            tuples_list = [(length, freq) for length, freq in zip(emojis_df['emojis'], emojis_df['percentage'])]
            emojis_distribution = str(tuples_list)
            context_dict[username]['emojis'] = emojis_distribution
            # print(emojis_distribution)
    
            # Should we do the percentage relative to all messages or to the total number of X column values (like it is now).
            # Ex. think about this: someone used only 2 emojies, while another person 20. 
    
        # STYLE KEYWORDS
        elif(file.find('mistral_style') != -1):
            df_mistral = pd.read_csv(file) 
            tuples_list = [(length, round(freq,2)) for length, freq in zip(df_mistral['keyword'], df_mistral['percentage'])]
            word_count_distribution = str(tuples_list)
            context_dict[username]['style_mistral'] = word_count_distribution
            # print('Mistral Style Keywords: ', word_count_distribution, '\n\n')
        elif(file.find('gpt_4_style') != -1):
            df_gpt = pd.read_csv(file) 
            tuples_list = [(length, round(freq,2)) for length, freq in zip(df_gpt['keyword'], df_gpt['percentage'])]
            word_count_distribution = str(tuples_list)
            context_dict[username]['style_gpt'] = word_count_distribution
        
        # SYNTAX KEYWORDS
        elif(file.find('mistral_syntax') != -1):
            df_mistral = pd.read_csv(file) 
            tuples_list = [(length, round(freq,2)) for length, freq in zip(df_mistral['keyword'], df_mistral['percentage'])]
            word_count_distribution = str(tuples_list)
            context_dict[username]['syntax_mistral'] = word_count_distribution
            # print('Mistral Style Keywords: ', word_count_distribution, '\n\n')
        elif(file.find('gpt_4_syntax') != -1):
            df_gpt = pd.read_csv(file) 
            tuples_list = [(length, round(freq,2)) for length, freq in zip(df_gpt['keyword'], df_gpt['percentage'])]
            word_count_distribution = str(tuples_list)
            context_dict[username]['syntax_gpt'] = word_count_distribution
            # print('GPT Style Keywords: ', word_count_distribution, '\n\n')
    
    df = pd.DataFrame.from_dict(context_dict, orient='index')
    df = df.reset_index().rename(columns={"index": "userID"})
    df.to_csv(output_context_folder + 'context_summary.csv')        
    
    #### Build the prompt context strings (per user)
    keys = ["U0","U1", "U2","U3","U4","U5","U6","U7","U8","U9"]
    for key in keys:
        prompt_context = '''Here are some details about the conversational style context of person X.  
        Use this information when infering the style of person X. The style context is given in a json format,  with the following metrics: word length distribution,  adjectives, functional words, emojis, style keywords, syntax keywords. For each of these metrics, we have extracted the top 5 most frequently used. For each metric,  the data is presented in a list of tuples as :(metric,frequency of occurences). Below is person X style context:{ '''
        p_context_mistral = ''
        p_context_gpt = ''
        # print(context_dict[key]['adjectives'])
        list_word_distribution = ast.literal_eval(context_dict[key]['word_distribution'])[0:5]
        prompt_context += 'word_distribution: ' + str(list_word_distribution) + ', \n'
        
        list_adjectives = ast.literal_eval(context_dict[key]['adjectives'])[0:5]
        prompt_context += 'adjectives: ' + str(list_adjectives) + ', \n'
        
        list_functional_words = ast.literal_eval(context_dict[key]['functional_words'])[0:5]
        prompt_context += 'functional_words: ' + str(list_functional_words) + ', \n'
    
        list_punctuations = ast.literal_eval(context_dict[key]['punctuations'])[0:5]
        prompt_context += 'punctuations: ' + str(list_punctuations) + ', \n'
        
        list_emojis = ast.literal_eval(context_dict[key]['emojis'])[0:5]
        prompt_context += 'emojis: ' + str(list_emojis) + ', \n'
    
        p_context_mistral = prompt_context
        p_context_gpt = prompt_context
        
        list_mistral_style = ast.literal_eval(context_dict[key]['style_mistral'])[0:5]
        p_context_mistral += 'style: ' + str(list_mistral_style) + ', \n'
    
        list_mistral_syntax = ast.literal_eval(context_dict[key]['syntax_mistral'])[0:5]
        p_context_mistral += 'syntax: ' + str(list_mistral_syntax) + ' \n'
    
        list_gpt_style = ast.literal_eval(context_dict[key]['style_gpt'])[0:5]
        p_context_gpt += 'style: ' + str(list_gpt_style) + ', \n'
       
        list_gpt_syntax = ast.literal_eval(context_dict[key]['syntax_gpt'])[0:5]
        p_context_gpt += 'syntax: ' + str(list_gpt_syntax) + ' \n'
    
        p_context_mistral+= ' }'
        p_context_gpt+= ' }'
    
        # print(list_word_distribution)
        # print(list_adjectives)
        # print(list_functional_words)
        # print(list_punctuations)
        # print(list_emojis)
        # print(list_mistral_style)
        # print(list_mistral_syntax)
        # print(list_gpt_style)
        # print(list_gpt_syntax)
        # print(p_context_mistral)
        # print(p_context_gpt)
        
        with open(output_context_folder + key + '_mistral_context.txt', "w") as f:
            f.write(p_context_mistral)
    
        with open(output_context_folder + key + '_gpt_context.txt', "w") as f:
            f.write(p_context_gpt)        
        
        
        
        

