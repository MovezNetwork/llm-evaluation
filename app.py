
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import production as p
import os
import inspect
import sys
import pandas as pd
import re
from datetime import datetime
import os
import math
import plotly.express as px
import configparser


# access parent directory from notebooks directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)



from dotenv import load_dotenv
load_dotenv()

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('LLM Output Details')
    add_vertical_space(6)



def main():
    # if you want to read input data from file uploader 
    # https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader

    def get_shots_data(text):
        # Define the pattern to match
        pattern = r'Here is the sentence written in the style of person X: (.+?)(?=\n\n)'

        # Extract sentences matching the pattern
        matches = re.findall(pattern, text, re.DOTALL)

        # Append matches to a single string with newlines
        result_string = '\n'.join(matches)

        return result_string
    
    def dataframe_with_selections_tst(df):
        df_with_selections = df.copy()
        df_with_selections.insert(0, "Display Details", False)

        # Get dataframe row-selections from user with st.data_editor
        edited_df = st.data_editor(
            df_with_selections,
            hide_index=True,
            column_config={"Display Details": st.column_config.CheckboxColumn(required=True)},
            disabled=df.columns,
        )

        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df['Display Details']]
        selection = selected_rows.drop('Display Details', axis=1)

        with st.sidebar:
            
            sidebar_width = st.sidebar.empty()._width
            for _, row in selection.iterrows():
                
                
                tst_id = row['tst_id'] 
                st.subheader('TST ID: ' + str(tst_id))
                
                st.write(row['neutral_sentence'])
                tst_sentence = row['tst_sentence']
                text_length_tst_sentence = len(tst_sentence)
                height_tst_sentence = max(100, (text_length_tst_sentence // 20 + 1) * 12)         
                tst_sentence_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_tst_sentence}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {tst_sentence}
                    </p>
                </div>
                """
                st.markdown(tst_sentence_html_code, unsafe_allow_html=True)
                st.text("")


                explanation = row['explanation']
                text_length_explanation = len(explanation)
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)         
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True)

                st.write(get_shots_data(row['query']))
                
                st.divider()
    def is_str_nan(s):
        try:
            return math.isnan(float(s))
        except ValueError:
            return False
    
    def dataframe_with_selections_eval(df):
        df_with_selections = df.copy()
        df_with_selections.insert(0, "Display Details", False)

        # Get dataframe row-selections from user with st.data_editor
        edited_df = st.data_editor(
            df_with_selections,
            hide_index=True,
            column_config={"Display Details": st.column_config.CheckboxColumn(required=True)},
            disabled=df.columns,
        )

        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df['Display Details']]
        selection = selected_rows.drop('Display Details', axis=1)

        with st.sidebar:
            
            sidebar_width = st.sidebar.empty()._width
            for _, row in selection.iterrows():
                
                
                tst_id = row['tst_id'] 
                st.subheader('TST ID: ' + str(tst_id))
                
                
                tst_sentence = row['tst_sentence']
                text_length_tst_sentence = len(tst_sentence)
                height_tst_sentence = max(100, (text_length_tst_sentence // 20 + 1) * 12)         
                tst_sentence_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_tst_sentence}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {tst_sentence}
                    </p>
                </div>
                """
                st.markdown(tst_sentence_html_code, unsafe_allow_html=True)
                st.text("")

                score_formality = row['eval_score_formality']
                st.subheader('Score Formality: ' + str(score_formality))
                explanation_score_formality = row['eval_explanation_formality']
                if explanation_score_formality is not None:
                    text_length_explanation = len(explanation_score_formality)
                else:
                    text_length_explanation = 5
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)       
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation_score_formality}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True)

                score_descriptiveness = row['eval_score_descriptiveness']
                st.subheader('Score Descriptiveness: ' + str(score_descriptiveness))
                explanation_score_descriptiveness = row['eval_explanation_descriptiveness']
          
                if not is_str_nan(explanation_score_descriptiveness):
                    text_length_explanation = len(explanation_score_descriptiveness)
                else:
                    text_length_explanation = 5
                    explanation_score_descriptiveness = "No explanation provided./Postprocessing error."
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)     
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation_score_descriptiveness}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True)               


                score_emotionality = row['eval_score_emotionality']
                st.subheader('Score Emotionality: ' + str(score_emotionality))
                explanation_score_emotionality = row['eval_explanation_emotionality']
                if not is_str_nan(explanation_score_emotionality):
                    text_length_explanation = len(explanation_score_emotionality)
                else:
                    text_length_explanation = 5
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)       
        
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation_score_emotionality}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True) 



                score_sentiment = row['eval_score_sentiment']
                st.subheader('Score Sentiment: ' + str(score_sentiment))
                explanation_score_sentiment = row['eval_explanation_sentiment']
                if not is_str_nan(explanation_score_sentiment):
                    text_length_explanation = len(explanation_score_sentiment)
                else:
                    text_length_explanation = 5
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)         
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation_score_sentiment}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True) 



                score_fluency = row['eval_score_fluency']
                score_comprehensibility = row['eval_score_comprehensibility']
                st.subheader('Score Fluency, Comprehensibility: ' + str(score_fluency) + ', ' + str(score_comprehensibility))
                explanation_score_fc = row['eval_explanation_fluency_comprehensibility']
                if not is_str_nan(explanation_score_fc):
                    text_length_explanation = len(explanation_score_fc)
                else:
                    text_length_explanation = 5
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)         
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation_score_fc}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True) 


                score_topic = row['eval_score_topic_similarity']
                st.subheader('Score Topic Similarity: ' + str(score_topic))
                explanation_score_topic = row['eval_explanation_topic_similarity']
                if not is_str_nan(explanation_score_topic):
                    text_length_explanation = len(explanation_score_topic)
                else:
                    text_length_explanation = 5
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)       
        
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation_score_topic}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True) 


                score_meaning = row['eval_score_meaning_similarity']
                st.subheader('Score Meaning Similarity: ' + str(score_meaning))
                explanation_score_meaning = row['eval_explanation_meaning_similarity']
                if not is_str_nan(explanation_score_meaning):
                    text_length_explanation = len(explanation_score_meaning)
                else:
                    text_length_explanation = 5
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)       
        
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation_score_meaning}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True) 

                st.divider()


    def dataframe_with_score(df,score):
        df_with_selections = df.copy()
        
        explanation_label = ''
        if score == 'fluency' or score == 'comprehensibility':
            explanation_label  = 'fluency_comprehensibility'
        else:
            explanation_label = score


        with st.sidebar:
            df_to_plot = df_eval_sub.groupby(['eval_score_' + score])['eval_score_' + score].count().reset_index(name='count')
            fig = px.bar(df_to_plot, x='eval_score_' + score, y='count', title='Score Distribution ' + score, labels={'eval_score_' + score: score +' Score', 'count':'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
            sidebar_width = st.sidebar.empty()._width
            for _, row in df_with_selections.iterrows():
                
                
                tst_id = row['tst_id'] 
                st.subheader('TST ID: ' + str(tst_id))
                
                
                tst_sentence = row['tst_sentence']
                text_length_tst_sentence = len(tst_sentence)
                height_tst_sentence = max(100, (text_length_tst_sentence // 20 + 1) * 12)         
                tst_sentence_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_tst_sentence}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {tst_sentence}
                    </p>
                </div>
                """
                st.markdown(tst_sentence_html_code, unsafe_allow_html=True)
                st.text("")

                if(score == 'topic_similarity' or score == 'meaning_similarity'):
                    #display the user sentence like the tst_sentence from above
                    st.subheader('User Rewritten Sentence:')

                    user_sentence = row['user_sentence']
                    text_length_user_sentence = len(user_sentence)
                    height_user_sentence = max(100, (text_length_user_sentence // 20 + 1) * 12)         
                    user_sentence_html_code = f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        width: {sidebar_width}px;
                        height: {height_user_sentence}px;
                        border-radius: 15px;
                        border: 1px solid black;
                        background-color: lightgrey;
                        margin: 0 auto;
                    ">
                        <p style="font-size: 15px;  color: black;">
                            {user_sentence}
                        </p>
                    </div>
                    """
                    st.markdown(user_sentence_html_code, unsafe_allow_html=True)
                    st.text("")

                score_metrics = row['eval_score_' + score]
                st.subheader('Score ' + score + ': ' + str(score_metrics))
                explanation_score_metrics = row['eval_explanation_'   + explanation_label]
                if not is_str_nan(explanation_score_metrics):
                    text_length_explanation = len(explanation_score_metrics)
                else:
                    text_length_explanation = 5
                height_explanation = max(100, (text_length_explanation // 20 + 1) * 12)       
                explanation_html_code = f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: {sidebar_width}px;
                    height: {height_explanation}px;
                    border-radius: 15px;
                    border: 1px solid black;
                    background-color: lightgrey;
                    margin: 0 auto;
                ">
                    <p style="font-size: 15px;  color: black;">
                        {explanation_score_metrics}
                    </p>
                </div>
                """
                st.markdown(explanation_html_code, unsafe_allow_html=True)

            

                st.divider()

    tab1, tab2, tab3 = st.tabs(["LLM Test Style Transfer", "LLM Evaluation", "New Runs"])

    def subdataframe_scores(df_eval,username_eval,sentence_id_eval):
        if username_eval and sentence_id_eval:
            df_eval = df_eval[(df_eval['username'] == username_eval) & (df_eval['id_neutral_sentence'] == sentence_id_eval)]
            # st.write('im in case 1')
        elif username_eval and not sentence_id_eval:
            df_eval = df_eval[(df_eval['username'] == username_eval)]
            # st.write('im in case 2')
        elif not username_eval and (sentence_id_eval or sentence_id_eval == 0):
            df_eval = df_eval[(df_eval['id_neutral_sentence'] == sentence_id_eval)]
            # st.write('im in case 3')
        else:
            df_eval = df_eval 
        
        return df_eval
    # TST tab
    with tab1:
        st.subheader("LLM TST Mistral Runs")
        # list all folders that starts with 'run_' found under f6_llm_tst_data directory
        folders = [name for name in os.listdir('f6_llm_tst_data') if os.path.isdir(os.path.join('f6_llm_tst_data', name)) and name.startswith('run_')]
        run_selection_llm_tst = st.selectbox("Select a llm tst run to display",folders,index=None, placeholder="Select LLM Run...")
        if run_selection_llm_tst:
            df_llm_tst_final = pd.read_csv('f6_llm_tst_data/'+ run_selection_llm_tst + '/'+ run_selection_llm_tst + '_tst_postprocess.csv')
            smallest_timestamp = datetime.strptime(str(df_llm_tst_final['timestamp'].min()), "%Y%m%d%H%M%S").strftime("%B %d %H:%M:%S")
            largest_timestamp = datetime.strptime(str(df_llm_tst_final['timestamp'].max()), "%Y%m%d%H%M%S").strftime("%B %d %H:%M:%S")
            st.write('Showing LLM TST Output generated between ' + smallest_timestamp + ' and ' + largest_timestamp)
        
            username = st.selectbox('User Selection', df_llm_tst_final['username'].unique(),index=None, placeholder="Select user...",)
            sentence_id = st.selectbox('Sentence Selection', df_llm_tst_final['id_neutral_sentence'].unique(),index=None, placeholder="Select sentence ID...",)
            
            if username and sentence_id:
                df_case1 = df_llm_tst_final[(df_llm_tst_final['username'] == username) & (df_llm_tst_final['id_neutral_sentence'] == sentence_id)]
                dataframe_with_selections_tst(df_case1)
            elif username and not sentence_id:
                df_case2 = df_llm_tst_final[(df_llm_tst_final['username'] == username)]
                dataframe_with_selections_tst(df_case2)
            elif not username and (sentence_id or sentence_id == 0):
                df_case3 = df_llm_tst_final[(df_llm_tst_final['id_neutral_sentence'] == sentence_id)]
                dataframe_with_selections_tst(df_case3)
            else:
                df_case4 = df_llm_tst_final 
                dataframe_with_selections_tst(df_case4)
        else:
            st.write('No data to display')
            
    # Evaluation tab
    with tab2:
        st.subheader("LLM Evaluation Runs")
        folders = [name for name in os.listdir('f6_llm_tst_data') if os.path.isdir(os.path.join('f6_llm_tst_data', name)) and name.startswith('run_')]
        run_selection_llm_eval = st.selectbox("Select a llm eval run to display",folders,index=None, placeholder="Select LLM Run...")
        if run_selection_llm_eval or run_selection_llm_tst:
            display_run = ''
            if run_selection_llm_tst:
                display_run = run_selection_llm_tst
            else:
                display_run = run_selection_llm_eval

            df_eval = pd.read_csv('f8_llm_evaluation_data/'+ display_run + '/postprocess_eval_'+ display_run + '.csv')
  

            username_eval = st.selectbox('User Selection ', df_eval['username'].unique(),index=None, placeholder="Select user...",)
            sentence_id_eval = st.selectbox('Sentence Selection ', df_eval['id_neutral_sentence'].unique(),index=None, placeholder="Select sentence ID...",)
            scores = ['Formality', 'Descriptiveness', 'Emotionality', 'Sentiment', 'Fluency','Comprehensibility', 'Topic Similarity', 'Meaning Similarity']
            eval_scores  = st.selectbox('Eval Metrics Selection ', scores,index=None, placeholder="Select evaluation metrics...",)

            df_eval_sub = pd.DataFrame()

            if eval_scores:

                    
                if eval_scores == 'Formality':
                    df_eval_sub = df_eval[[ 'tst_id','username','id_neutral_sentence','tst_sentence','eval_score_formality','eval_explanation_formality']]
                    df_eval_sub = subdataframe_scores(df_eval_sub,username_eval,sentence_id_eval)
                    dataframe_with_score(df_eval_sub,'formality')
                    st.dataframe(df_eval_sub)               

                elif eval_scores == 'Descriptiveness':
                    df_eval_sub = df_eval[[ 'tst_id','username','id_neutral_sentence','tst_sentence','eval_score_descriptiveness','eval_explanation_descriptiveness']]
                    df_eval_sub = subdataframe_scores(df_eval_sub,username_eval,sentence_id_eval)
                    dataframe_with_score(df_eval_sub,'descriptiveness')
                    st.dataframe(df_eval_sub)               

                elif eval_scores == 'Emotionality':
                    df_eval_sub = df_eval[[ 'tst_id','username','id_neutral_sentence','tst_sentence','eval_score_emotionality','eval_explanation_emotionality']]
                    df_eval_sub = subdataframe_scores(df_eval_sub,username_eval,sentence_id_eval)
                    dataframe_with_score(df_eval_sub,'emotionality')
                    st.dataframe(df_eval_sub)               
    
                elif eval_scores == 'Sentiment':
                    df_eval_sub = df_eval[[ 'tst_id','username','id_neutral_sentence','tst_sentence','eval_score_sentiment','eval_explanation_sentiment']]
                    df_eval_sub = subdataframe_scores(df_eval_sub,username_eval,sentence_id_eval)
                    dataframe_with_score(df_eval_sub,'sentiment')
                    st.dataframe(df_eval_sub)               

                elif eval_scores == 'Fluency':
                    df_eval_sub = df_eval[[ 'tst_id','username','id_neutral_sentence','tst_sentence','eval_score_fluency','eval_explanation_fluency_comprehensibility']]
                    df_eval_sub = subdataframe_scores(df_eval_sub,username_eval,sentence_id_eval)
                    dataframe_with_score(df_eval_sub,'fluency')
                    st.dataframe(df_eval_sub)               

                elif eval_scores == 'Comprehensibility':
                    df_eval_sub = df_eval[[ 'tst_id','username','id_neutral_sentence','tst_sentence','eval_score_comprehensibility','eval_explanation_fluency_comprehensibility']]
                    df_eval_sub = subdataframe_scores(df_eval_sub,username_eval,sentence_id_eval)
                    dataframe_with_score(df_eval_sub,'comprehensibility')
                    st.dataframe(df_eval_sub)     

                
                elif eval_scores == 'Topic Similarity':
                    df_eval_sub = df_eval[[ 'tst_id','username','user_sentence','id_neutral_sentence','tst_sentence','eval_score_topic_similarity','eval_explanation_topic_similarity']]
                    df_eval_sub = subdataframe_scores(df_eval_sub,username_eval,sentence_id_eval)
                    dataframe_with_score(df_eval_sub,'topic_similarity')
                    st.dataframe(df_eval_sub) 

                elif eval_scores == 'Meaning Similarity':
                    df_eval_sub = df_eval[[ 'tst_id','username','user_sentence','id_neutral_sentence','tst_sentence','eval_score_meaning_similarity','eval_explanation_meaning_similarity']]
                    df_eval_sub = subdataframe_scores(df_eval_sub,username_eval,sentence_id_eval)
                    dataframe_with_score(df_eval_sub,'meaning_similarity')
                    st.dataframe(df_eval_sub)               


            if df_eval_sub.empty:
                if username_eval and sentence_id_eval:
                    df_case1 = df_eval[(df_eval['username'] == username_eval) & (df_eval['id_neutral_sentence'] == sentence_id_eval)]
                    dataframe_with_selections_eval(df_case1)
                    # st.write('im in case 1')
                elif username_eval and not sentence_id_eval:
                    df_case2 = df_eval[(df_eval['username'] == username_eval)]
                    dataframe_with_selections_eval(df_case2)
                    # st.write('im in case 2')
                elif not username_eval and (sentence_id_eval or sentence_id_eval == 0):
                    df_case3 = df_eval[(df_eval['id_neutral_sentence'] == sentence_id_eval)]
                    dataframe_with_selections_eval(df_case3)
                    # st.write('im in case 3')
                else:
                    df_case4 = df_eval 
                    dataframe_with_selections_eval(df_case4)

        else:
            st.write('No data to display')


    # New Runs tab
    with tab3:
        st.subheader("New LLM Run")
        models = ['mistral-small', 'mistral-medium','gpt-3.5-turbo-0125','gpt-4-turbo','gpt-4o-mini','gpt-4o']
        prompts = ['k-shot non-parallel','k-shot parallel']


        model_selection = st.selectbox("Select a LLM",models,index=None, placeholder="Select LLM Model...")
        if model_selection:
            st.write('Selected model',  model_selection)

            prompt_selection = st.selectbox("Select a prompt",prompts,index=None, placeholder="Select prompt...")
            
            config = configparser.ConfigParser()
            # Read the configuration file & paths
            config.read('config.ini')
            surfdrive_url_prompts = config.get('credentials', 'surfdrive_url_prompts')

            if prompt_selection == 'k-shot non-parallel':
                prompt_id = 2
                # reading prompt template components - depends on prompt_id
                df_prompts = pd.read_csv(surfdrive_url_prompts,sep=';').reset_index()
                mistral_prompt_system_content = df_prompts['prompt_system_content'].iloc[prompt_id]
                mistral_prompt_x_shot_template = df_prompts['prompt_x_shot_template'].iloc[prompt_id]
                mistral_prompt_content_addition = df_prompts['prompt_content_addition'].iloc[prompt_id]

                system_prompt = st.text_area(
                    "System Prompt:",
                    mistral_prompt_system_content,
                )
                task_prompt = st.text_area(
                    "Task Prompt:",
                   mistral_prompt_x_shot_template,
                )
                inference_prompt = st.text_area(
                    "Inference Prompt:",
                    mistral_prompt_content_addition,
                )
            
            elif prompt_selection == 'k-shot parallel':
                prompt_id = 0
                # reading prompt template components - depends on prompt_id
                df_prompts = pd.read_csv(surfdrive_url_prompts,sep=';').reset_index()
                mistral_prompt_system_content = df_prompts['prompt_system_content'].iloc[prompt_id]
                mistral_prompt_x_shot_template = df_prompts['prompt_x_shot_template'].iloc[prompt_id]
                mistral_prompt_content_addition = df_prompts['prompt_content_addition'].iloc[prompt_id]

                system_prompt = st.text_area(
                    "System Prompt:",
                    mistral_prompt_system_content,
                )
                task_prompt = st.text_area(
                    "K-shot Examples Prompt:",
                   mistral_prompt_x_shot_template,
                )
                inference_prompt = st.text_area(
                    "Inference Prompt:",
                    mistral_prompt_content_addition,
                )

            users = st.multiselect(
                "Which users to include?",
                ["BU01", "BU02", "BU03", "BU04","BU05", "BU06", "BU07", "BU08"],
                ["BU01", "BU02", "BU03", "BU04","BU05", "BU06", "BU07", "BU08"],
            )
            sentences = st.multiselect(
                "Which sentences to include?",
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
            )

            if st.button('Start LLM TST'):
                # open dialog window in streamlit to confirm the run

            
                st.write('Running LLM Text Style Transfer')

                df_user_data,neutral_sentences,user_sentences = p.read_input_data()
                df_user_data = df_user_data[df_user_data['username'].isin(users)]
                neutral_sentences = neutral_sentences[neutral_sentences['sentenceid'].isin(sentences)]

                df_llm_tst,output_run = p.llm_tst(df_user_data,neutral_sentences,model_selection,system_prompt,task_prompt,inference_prompt,prompt_id)
                st.write(df_llm_tst)
                st.write('Postprocess LLM TST')

                df_llm_tst_processed = p.postprocess_llm_tst(df_llm_tst,output_run)
                st.write(df_llm_tst_processed)

                st.write('Evaluate LLM TST')
                df_eval = p.llm_evl(df_llm_tst_processed,user_sentences,model_selection)
                st.write(df_eval)

                # # check if df is not empty
                # if not df_llm_tst.empty:
                #     st.write(df_llm_tst)
                #     if st.button('Start LLM EVAL'):
                #         st.write('Running LLM-based Evaluation')
                #         try:
                #             st.write(df_eval)
                #         except:
                #             st.write('Error in LLM Evaluation') 
                st.write('New LLM RUN Generated with run_id: ', df_llm_tst['output_run'].unique()[0])
        
        else:
            st.write('')














if __name__ == '__main__':

    main() 