import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import time
import re

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

def validate_jsonl(file_path):
    """Validate JSONL file structure and content with enhanced checks"""
    errors = []
    required_labels = {'Positive', 'Negative', 'Neutral'}
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # Structure validation
                if "messages" not in data:
                    errors.append(f"Line {i}: Missing 'messages' key")
                    continue
                    
                messages = data["messages"]
                roles = [msg["role"] for msg in messages]
                contents = [msg["content"] for msg in messages]
                
                # Role validation
                if "system" not in roles:
                    errors.append(f"Line {i}: Missing system message")
                if "user" not in roles:
                    errors.append(f"Line {i}: Missing user message")
                if "assistant" not in roles:
                    errors.append(f"Line {i}: Missing assistant message")
                
                # Content validation
                if len(messages) != 3:
                    errors.append(f"Line {i}: Incorrect message count ({len(messages)} instead of 3)")
                
                assistant_content = next(msg["content"] for msg in messages if msg["role"] == "assistant")
                if assistant_content not in required_labels:
                    errors.append(f"Line {i}: Invalid label '{assistant_content}'")
                
                # Text formatting validation
                user_content = next(msg["content"] for msg in messages if msg["role"] == "user")
                if re.search(r",\s*(?=[A-Z0-9])", user_content):  # Check for CSV-like patterns
                    errors.append(f"Line {i}: Detected CSV artifacts in user content")
                if '"' in user_content:
                    errors.append(f"Line {i}: Unnecessary quotes in user content")
                    
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {str(e)}")
    
    return errors

# Streamlit UI
st.title("End-to-End Sentiment Analysis Fine-tuning Demo")

# Section 1: Initial Analysis
with st.expander("Step 1: Compare GPT-4 and GPT-3.5 Results", expanded=True):
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            text_cols = [col for col in df.columns if 'feedback' in col]
            
            if not text_cols:
                st.error("No text column found")
                st.stop()
                
            text_col = text_cols[0]
            
            if st.button("Run Initial Analysis"):
                with st.spinner("Analyzing with GPT-4 and GPT-3.5..."):
                    # Clean text inputs before analysis
                    df['clean_text'] = df[text_col].apply(
                        lambda x: x.split('",')[0].replace('"', '').strip()
                    )
                    
                    df['gpt4'] = df['clean_text'].apply(lambda x: get_sentiment('gpt-4', x))
                    df['gpt35'] = df['clean_text'].apply(lambda x: get_sentiment('gpt-3.5-turbo', x))
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
                # Extract clean text from original column
                clean_text = row[text_col].split('",')[0].replace('\"', '').strip()
                
                training_data.append({
                    "messages": [
                        {"role": "system", "content": "Classify sentiment as Positive, Negative, or Neutral. Respond only with the label."},
                        {"role": "user", "content": clean_text},
                        {"role": "assistant", "content": row['gpt4']}
                    ]
                })

            # Add deduplication here
            seen = set()
            training_data = [
                d for d in training_data 
                if not (
                    (d['messages'][1]['content'], d['messages'][2]['content']) in seen 
                    or seen.add((d['messages'][1]['content'], d['messages'][2]['content']))
                )
            ]

            with open('training_data.jsonl', 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            # Validate JSONL file
            validation_errors = validate_jsonl('training_data.jsonl')
            if validation_errors:
                st.error("Validation errors found:")
                for error in validation_errors:
                    st.write(error)
                st.stop()
            
            st.session_state.training_file = 'training_data.jsonl'
            st.success("Training data generated!")
            st.download_button("Download Training Data", open('training_data.jsonl').read(), "training.jsonl")

# Section 3: Fine-tuning
if 'training_file' in st.session_state:
    with st.expander("Step 3: Fine-tune GPT-3.5"):
        if st.button("Start Fine-tuning Job"):
            with st.spinner("Uploading training file and starting job..."):
                try:
                    # Upload training file
                    file_obj = client.files.create(
                        file=open(st.session_state.training_file, "rb"),
                        purpose="fine-tune"
                    )
                    
                    # Wait for file to process
                    while True:
                        file_status = client.files.retrieve(file_obj.id).status
                        if file_status == "processed":
                            break
                        time.sleep(1)
                    
                    # Start fine-tuning job with latest model version
                    job = client.fine_tuning.jobs.create(
                        training_file=file_obj.id,
                        model="gpt-3.5-turbo-0125",  # Explicitly specify model version
                        suffix="sentiment-ft",
                        hyperparameters={"n_epochs": 3}
                    )
                    
                    st.session_state.job_id = job.id
                    st.session_state.ft_model = None
                    st.success(f"Fine-tuning job started! ID: {job.id}")
                
                except Exception as e:
                    st.error(f"Fine-tuning failed: {str(e)}")
                    if hasattr(e, 'response') and hasattr(e.response, 'json'):
                        error_details = e.response.json().get('error', {})
                        st.write("Error details:", error_details)

# Section 4: Monitor Fine-tuning
if 'job_id' in st.session_state:
    with st.expander("Step 4: Monitor Fine-tuning Progress"):
        if st.button("Check Status"):
            try:
                job = client.fine_tuning.jobs.retrieve(st.session_state.job_id)
                st.write(f"Status: {job.status}")
                
                # Show detailed error if available
                if job.status == 'failed' and job.error:
                    # Safely print error details
                    if isinstance(job.error, dict):
                        st.error(f"Error: {job.error.get('message', str(job.error))}")
                        if 'code' in job.error:
                            st.write(f"Error code: {job.error['code']}")
                    else:
                        st.error(f"Error: {str(job.error)}")
                
                # Show job events (fixed parameter passing)
                st.subheader("Job Events")
                events = client.fine_tuning.jobs.list_events(
                    st.session_state.job_id,  # Positional argument first
                    limit=20
                )
                for event in reversed(list(events)):
                    st.write(f"{event.created_at}: {event.message}")
                
                if job.fine_tuned_model:
                    st.session_state.ft_model = job.fine_tuned_model
                    st.success(f"Model ready! Name: {job.fine_tuned_model}")
            
            except Exception as e:
                st.error(f"Failed to retrieve job status: {str(e)}")

# Section 5: Test Fine-tuned Model
if 'ft_model' in st.session_state:
    with st.expander("Step 5: Test Fine-tuned Model"):
        if st.button("Run Comparative Analysis"):
            with st.spinner("Testing fine-tuned model..."):
                try:
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
                
                except Exception as e:
                    st.error(f"Model access failed: {str(e)}")
                    st.write("Ensure:")
                    st.write("- You're using the correct API key")
                    st.write("- Model ID is correct")
                    st.write("- Model has finished deployment")
