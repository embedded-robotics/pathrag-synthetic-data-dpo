import pandas as pd
import os
import json
import openai
import backoff
from PIL import Image
import pickle
import time

file_path = os.path.join("data", "pubmed_set", "captions.json")

with open(file_path, 'rb') as file:
    captions_data = json.load(file)
    
OPENAI_API_PATH = os.path.join(os.getcwd(), 'api.key')

with open(OPENAI_API_PATH) as f:
    openai.api_key = f.read().strip()

@backoff.on_exception(backoff.expo, openai.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.chat.completions.create(**kwargs)

def gpt(user_prompt, system_prompt="You are an expert pathologist", model="gpt-4", temperature=0.7, max_tokens=1000) -> list:

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    
    return res.choices[0].message.content

base_prompt = '''You are provided with a text description (figure caption) of a pathology image. Unfortunately, you don't have access to the original image.
Your job is to generate a total of 5 open-ended question/answer pairs from this figure caption starting with "What" or "Where". Below are the requirements to generate the question/answer pairs:

- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers or names, as these may reveal the conversation is based on the text information, rather than image itself.
- Focus on the visual aspects of the image that can be inferred without the text information
- Do not use phrases like "mentioned", "caption", "context", "without the image" in the question/answer pairs. Instead, refer to the information as being "in the image" or preferably don't mention anything
- Ensure that question/anwer pairs are diverse and cover a range of visual aspects of the image
- Answer responsibly, avoiding overconfidence, and do not provide medical advice or diagnostic information

Caption: {caption}
Question:
Answer:
'''

# Getting the results and saving it
index_list = []
caption_list = []
uuid_list = []
llm_response_list = []

start_index = 0
current_index = start_index
total_records = len(captions_data)

while True:
    try:
        for index in range(start_index, total_records):
            current_index = index
            caption = captions_data[str(current_index)]['caption']
            uuid = captions_data[str(current_index)]['uuid']
            
            user_prompt = base_prompt.format(caption = caption)
            response = gpt(user_prompt)
            
            index_list.append(current_index)
            caption_list.append(caption)
            uuid_list.append(uuid)
            llm_response_list.append(response)

            print("Index:", current_index)
            print("Caption:", caption)
            print("UUID:", uuid)
            print()
            print(response)
            print()
    
    except Exception as err:
        print("Something went wrong: ", err)
        start_index = current_index
        print("Waiting for 10 seconds before continuing again with index:", start_index)
        time.sleep(10)

    # Break the loop if current_index has completed
    if current_index == (total_records - 1):
        break


llm_qa_pairs = pd.DataFrame({'index': index_list, 'caption': caption_list, 'uuid': uuid_list, 'llm_qa_pairs': llm_response_list})

file_name = 'llm_qa_pairs_' + str(start_index) + '_' + str(total_records) + '.pkl'

with open(file_name, 'wb') as file:
    pickle.dump(llm_qa_pairs, file)