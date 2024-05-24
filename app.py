
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import production as p
import os
import inspect
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import os

# access parent directory from notebooks directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)



from dotenv import load_dotenv
load_dotenv()

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('LLM TST Study')
    add_vertical_space(6)



def main():
    # if you want to read input data from file uploader 
    # https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader

    def dataframe_with_selections(df):
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

                st.divider()
        add_vertical_space(4)

    tab1, tab2 = st.tabs(["LLM Test Style Transfer", "LLM Evaluation"])

    with tab1:
        st.subheader("LLM TST Mistral Runs")

  
        if st.button('Run new TST'):  # Returns True if the user clicks the button
            st.write('Running Text Style Transfer...')
            df_user_data,neutral_sentences = p.read_input_data()
            df_llm_tst = p.llm_tst(df_user_data,neutral_sentences[0:2])
            df_llm_tst_final = p.postprocess_llm_tst(df_llm_tst)

        df_llm_tst_final = p.read_and_postprocess_llm_tst()
        smallest_timestamp = datetime.strptime(str(df_llm_tst_final['timestamp'].min()), "%Y%m%d%H%M%S").strftime("%B %d %H:%M:%S")
        largest_timestamp = datetime.strptime(str(df_llm_tst_final['timestamp'].max()), "%Y%m%d%H%M%S").strftime("%B %d %H:%M:%S")
        st.write('Showing LLM TST Output generated between ' + smallest_timestamp + ' and ' + largest_timestamp)
        
        username = st.selectbox('User Selection', df_llm_tst_final['username'].unique(),index=None, placeholder="Select user...",)
        sentence_id = st.selectbox('Sentence Selection', df_llm_tst_final['id_neutral_sentence'].unique(),index=None, placeholder="Select sentence ID...",)
        if username and sentence_id:
            df_case1 = df_llm_tst_final[(df_llm_tst_final['username'] == username) & (df_llm_tst_final['id_neutral_sentence'] == sentence_id)]
            dataframe_with_selections(df_case1)
            # st.write('im in case 1')
        elif username and not sentence_id:
            df_case2 = df_llm_tst_final[(df_llm_tst_final['username'] == username)]
            dataframe_with_selections(df_case2)
            # st.write('im in case 2')
        elif not username and sentence_id:
            df_case3 = df_llm_tst_final[(df_llm_tst_final['id_neutral_sentence'] == sentence_id)]
            dataframe_with_selections(df_case3)
            # st.write('im in case 3')
        else:
            df_case4 = df_llm_tst_final 
            dataframe_with_selections(df_case4)
            # st.write('im in case 4')
            






     

    with tab2:
        st.subheader("LLM Evaluation Runs")
        # Sample DataFrame
        data = {
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['foo', 'bar', 'baz', 'qux', 'quux']
        }
        df = pd.DataFrame(data)

        # Display the DataFrame
        st.dataframe(df)

        # Allow the user to select a row
        selected_row = st.selectbox("Select a row:", df.index)

        # Get the selected row's data
        selected_data = df.loc[selected_row]

        # Define the HTML and CSS for the square with dynamic text
        html_code = f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            width: 200px;
            height: 200px;
            border: 2px solid black;
            background-color: lightgrey;
            margin: 0 auto;
        ">
            <p style="font-size: 20px; font-weight: bold; color: black;">
                Column A: {selected_data['A']}<br>
                Column B: {selected_data['B']}<br>
                Column C: {selected_data['C']}
            </p>
        </div>
        """

        # Use st.markdown to display the HTML
        st.markdown(html_code, unsafe_allow_html=True)







if __name__ == '__main__':

    main() 