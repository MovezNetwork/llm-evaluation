import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML

def get_input_data():
    output_evaluation_folder_path = 'f8_llm_evaluation_data/'
    output_processed_evaluation_folder_path = 'f9_processed_llm_evaluation_data/'
    mistral_noContext = pd.read_csv(output_processed_evaluation_folder_path + 'Evaluation_NoContext_mistral-medium_corrected.csv')
    mistral_context = pd.read_csv(output_processed_evaluation_folder_path + 'Evaluation_Context_mistral-medium_corrected.csv')
    gpt_noContext = pd.read_csv(output_processed_evaluation_folder_path + 'Evaluation_NoContext_gpt-4_corrected.csv')
    gpt_context = pd.read_csv(output_processed_evaluation_folder_path + 'Evaluation_Context_gpt-4_corrected.csv')
    mistral_noContext['evaluator'] = 'mistral-medium'
    gpt_noContext['evaluator'] = 'gpt-4'
    mistral_context['evaluator'] = 'mistral-medium'
    gpt_context['evaluator'] = 'gpt-4'
    
    is_parallel_map = {0: True, 1: True, 2: False, 3: False}
    mistral_noContext['isParallel'] = mistral_noContext['promptID'].map(is_parallel_map)
    gpt_noContext['isParallel'] = gpt_noContext['promptID'].map(is_parallel_map)
    mistral_context['isParallel'] = mistral_context['promptID'].map(is_parallel_map)
    gpt_context['isParallel'] = gpt_context['promptID'].map(is_parallel_map)
    
    df_noContext = pd.concat([gpt_noContext, mistral_noContext], axis=0)
    df_context = pd.concat([gpt_context, mistral_context], axis=0)
    df_all = pd.concat([df_noContext, df_context], axis=0)
    
    return df_all

def display_interactive_dataframe(df):
    # Create dropdown widgets for each column you want to filter on
    user_dropdown = widgets.Dropdown(options=['All'] + df['user'].unique().tolist(), value = df['user'].unique().tolist()[0], description='User:')
    promptID_dropdown = widgets.Dropdown(options=['All'] + df['promptID'].unique().tolist(), value = df['promptID'].unique().tolist()[0],description='Prompt ID:')
    model_dropdown = widgets.Dropdown(options=['All'] + df['model'].unique().tolist(), value = df['model'].unique().tolist()[0], description='Model:')
    shots_dropdown = widgets.Dropdown(options=['All'] + df['shots'].unique().tolist(), value = df['shots'].unique().tolist()[0], description='Shots:')
    isParallel_dropdown = widgets.Dropdown(options=['All', True, False], value = True, description='Is Parallel:')
    prompting_dropdown = widgets.Dropdown(options=['All'] + df['prompting'].unique().tolist(), value = df['prompting'].unique().tolist()[0], description='Prompting:')
    evaluator_dropdown = widgets.Dropdown(options=['All'] + df['evaluator'].unique().tolist(), value = df['evaluator'].unique().tolist()[0], description='Evaluator:')
    score_dropdown = widgets.Dropdown(options=['None', 'Accuracy', 'Content_Preservation', 'Fluency'], description='Score:')

    # Create function to update displayed data based on dropdown selections
    def update_data(user, promptID, model, shots, prompting, evaluator, isParallel, score_type):
        filtered_df = df.copy()
        if user != 'All':
            filtered_df = filtered_df[filtered_df['user'] == user]
        if promptID != 'All':
            filtered_df = filtered_df[filtered_df['promptID'] == promptID]
        if model and model != 'All':
            filtered_df = filtered_df[filtered_df['model'] == model]
        if shots != 'All':
            filtered_df = filtered_df[filtered_df['shots'] == shots]
        if prompting != 'All':
            filtered_df = filtered_df[filtered_df['prompting'] == prompting]
        if evaluator != 'All':
            filtered_df = filtered_df[filtered_df['evaluator'] == evaluator]
        if isParallel != 'All':
            filtered_df = filtered_df[filtered_df['isParallel'] == isParallel]

        # Display columns based on score type selection
        if score_type == 'None':
            display_data = filtered_df[['original', 'rewritten_sentence','your_text']]
        elif score_type in ['Accuracy', 'Content_Preservation', 'Fluency']:
            explanation_column = f'explanation_{score_type.lower()}'
            display_data = filtered_df[['original', 'rewritten_sentence','your_text', explanation_column]]
        else:
            print("Invalid score type selected. Displaying original and rewritten_sentence columns only.")
            display_data = filtered_df[['original', 'rewritten_sentence','your_text']]

        # HTML table styling
        html_table = display_data.to_html(index=False)
        styled_html_table = "<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #dddddd; text-align: left; padding: 8px;} th {background-color: #f2f2f2;} tr:nth-child(even) {background-color: #f2f2f2;}</style>" + html_table

        display(HTML(styled_html_table))


    display(widgets.interactive(update_data, 
                                user=user_dropdown, 
                                promptID=promptID_dropdown, 
                                model=model_dropdown, 
                                shots=shots_dropdown, 
                                prompting=prompting_dropdown, 
                                evaluator=evaluator_dropdown, 
                                isParallel=isParallel_dropdown,
                                score_type=score_dropdown))

        