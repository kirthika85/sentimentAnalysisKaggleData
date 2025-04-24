import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import time

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_sentiment(model: str, text: str) -> str:
    """Get sentiment classification from specified model"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Classify sentiment as Positive, Negative, or Neutral. Respond only with the label."},
                {"role": "user", "content": str(text)}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("End-to-End Sentiment Analysis Fine-tuning Demo")

# Section 1: Initial Analysis
with st.expander("Step 1: Compare GPT-4 and GPT-3.5 Results", expanded=True):
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            text_cols = [col for col in df.columns if 'text' in col]
            
            if not text_cols:
                st.error("No text column found")
                st.stop()
                
            text_col = text_cols[0]
            
            if st.button("Run Initial Analysis"):
                with st.spinner("Analyzing with GPT-4 and GPT-3.5..."):
                    df['gpt4'] = df[text_col].apply(lambda x: get_sentiment('gpt-4', x))
                    df['gpt35'] = df[text_col].apply(lambda x: get_sentiment('gpt-3.5-turbo', x))
                    df['discrepancy'] = df['gpt4'] != df['gpt35']
                
                st.session_state.df = df
                st.success("Analysis complete!")
                st.write("Discrepancies found:", df['discrepancy'].sum())
                st.dataframe(df)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Section 2: Prepare Training Data
if 'df' in st.session_state and st.session_state.df['discrepancy'].sum() > 0:
    with st.expander("Step 2: Prepare Fine-tuning Data"):
        discrepant_df = st.session_state.df[st.session_state.df['discrepancy']]
        
        if st.button("Generate Training JSONL"):
            training_data = []
            for _, row in discrepant_df.iterrows():
                training_data.append({
                    "messages": [
                        {"role": "system", "content": "Classify sentiment as Positive, Negative, or Neutral. Respond only with the label."},
                        {"role": "user", "content": row[text_col]},
                        {"role": "assistant", "content": row['gpt4']}
                    ]
                })
            
            with open('training_data.jsonl', 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            st.session_state.training_file = 'training_data.jsonl'
            st.success("Training data generated!")
            st.download_button("Download Training Data", open('training_data.jsonl').read(), "training.jsonl")

# Section 3: Fine-tuning
if 'training_file' in st.session_state:
    with st.expander("Step 3: Fine-tune GPT-3.5"):
        if st.button("Start Fine-tuning Job"):
            with st.spinner("Uploading training file and starting job..."):
                # Upload training file
                file_obj = client.files.create(
                    file=open(st.session_state.training_file, "rb"),
                    purpose="fine-tune"
                )
                
                # Start fine-tuning job
                job = client.fine_tuning.jobs.create(
                    training_file=file_obj.id,
                    model="gpt-3.5-turbo",
                    suffix="sentiment-ft"
                )
                
                st.session_state.job_id = job.id
                st.session_state.ft_model = None
                st.success(f"Fine-tuning job started! ID: {job.id}")

# Section 4: Monitor Fine-tuning
if 'job_id' in st.session_state:
    with st.expander("Step 4: Monitor Fine-tuning Progress"):
        if st.button("Check Status"):
            job = client.fine_tuning.jobs.retrieve(st.session_state.job_id)
            st.write(f"Status: {job.status}")
            
            if job.fine_tuned_model:
                st.session_state.ft_model = job.fine_tuned_model
                st.success(f"Model ready! Name: {job.fine_tuned_model}")

# Section 5: Test Fine-tuned Model
if 'ft_model' in st.session_state:
    with st.expander("Step 5: Test Fine-tuned Model"):
        if st.button("Run Comparative Analysis"):
            with st.spinner("Testing fine-tuned model..."):
                df = st.session_state.df
                df['ft_gpt35'] = df[text_col].apply(lambda x: get_sentiment(st.session_state.ft_model, x))
                df['new_discrepancy'] = df['gpt4'] != df['ft_gpt35']
                
                st.success("Post-training analysis complete!")
                st.write("Remaining discrepancies:", df['new_discrepancy'].sum())
                st.dataframe(df[['text', 'gpt4', 'gpt35', 'ft_gpt35']])
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Results",
                    data=csv,
                    file_name='final_results.csv',
                    mime='text/csv'
                )
