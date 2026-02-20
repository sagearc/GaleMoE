import pandas as pd
import time
import json
import asyncio
from google import genai
from pydantic import BaseModel, Field
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "results/neuron_top20_titles_layer16_expert0_w3.csv"
OUTPUT_FILE = "clustered_neurons.json"
BATCH_SIZE = 128 
MAX_RETRIES = 5
PARALLEL_REQUESTS = 112

client = genai.Client()

# --- DEFINE PYDANTIC SCHEMAS ---
class TopicDetail(BaseModel):
    domain: str = Field(description="The broad, general category (e.g., 'Biology', 'Geography', 'Sports').")
    specific_niche: str = Field(description="The highly granular, specific unifying theme. MUST be specific (e.g., 'Moth Species', 'Rivers in Eastern Europe', '1990s Italian Football').")

class NeuronTopic(BaseModel):
    neuron_id: int = Field(description="The exact ID of the neuron from the input data.")
    topics: list[TopicDetail] = Field(description="Exactly TWO topic classifications for these titles.")

class BatchResult(BaseModel):
    results: list[NeuronTopic] = Field(description="A list of processed neurons.")

async def get_topics_from_gemini(batch_df):
    """Sends a batch to Gemini and returns parsed JSON results using Structured Outputs."""
    
    batch_data = [
        {"id": int(row['neuron_id']), "titles": row.iloc[1:].dropna().tolist()}
        for _, row in batch_df.iterrows()
    ]

    prompt = f"""
    You are an expert data taxonomist. Analyze the following list of neurons and their associated Wikipedia titles.
    For each neuron ID, identify TWO distinct themes that unify the titles. 
    
    CRITICAL INSTRUCTIONS FOR 'specific_niche':
    - NEVER use broad, generic single words like "Animals", "Geography", "History", or "Sports".
    - You MUST be highly specific and granular.
    - BAD niche: "Animals" -> GOOD niche: "Moth Species"
    - BAD niche: "Geography" -> GOOD niche: "Municipalities in Denmark"
    - BAD niche: "History" -> GOOD niche: "World War II Naval Battles"
    - BAD niche: "Sports" -> GOOD niche: "2020-21 European Ice Hockey Seasons"

    Data:
    {json.dumps(batch_data)}
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": BatchResult.model_json_schema(),
                    "temperature": 0.1 
                }
            )
            
            # Validate and parse the response directly into our Pydantic model
            parsed_data = BatchResult.model_validate_json(response.text)
            
            # Convert back to standard dictionaries for easy saving
            return [item.model_dump() for item in parsed_data.results]
            
        except Exception as e:
            wait_time = (attempt + 1) * 5 
            print(f"Error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"FAILED batch after {MAX_RETRIES} attempts. Skipping.")
                return [] 

# --- MAIN EXECUTION ---
async def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # Load existing results if resuming
    try:
        with open(OUTPUT_FILE, "r") as f:
            all_results = json.load(f)
            processed_ids = {item["neuron_id"] for item in all_results}
            print(f"Found {len(processed_ids)} already processed rows. Resuming...")
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []
        processed_ids = set()

    # Filter out already processed rows
    df_to_process = df[~df['neuron_id'].isin(processed_ids)]
    print(f"Rows remaining to process: {len(df_to_process)}")

    # Create list of all batches
    all_batches = []
    for i in range(0, len(df_to_process), BATCH_SIZE):
        batch_df = df_to_process.iloc[i : i + BATCH_SIZE]
        all_batches.append((i, batch_df))
    
    # Process batches in parallel groups
    with tqdm(total=len(all_batches), desc="Processing Batches", unit="batch") as pbar:
        for chunk_start in range(0, len(all_batches), PARALLEL_REQUESTS):
            chunk_end = min(chunk_start + PARALLEL_REQUESTS, len(all_batches))
            batch_chunk = all_batches[chunk_start:chunk_end]
            
            print(f"\nProcessing {len(batch_chunk)} batches in parallel...")
            
            # Run parallel requests
            tasks = [get_topics_from_gemini(batch_df) for _, batch_df in batch_chunk]
            chunk_results = await asyncio.gather(*tasks)
            
            # Extend results from all parallel requests
            for batch_results in chunk_results:
                if batch_results:
                    all_results.extend(batch_results)
            
            # Save progress incrementally after each parallel chunk
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            
            pbar.update(len(batch_chunk))

    print(f"\nTask Complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
