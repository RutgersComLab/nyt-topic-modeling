import pickle
import torch
import torch.nn as nn
from loguru import logger
from pytorch_transformers import BertTokenizer
import os
from pytorch_transformers import BertConfig, BertForSequenceClassification
from tqdm import tqdm

def split_input_sequence(input_sequence, max_length, tokenizer):
    input_ids = tokenizer.encode(
        input_sequence,
        add_special_tokens=False,
    )

    segments = []
    current_segment = []

    for token_id in input_ids:
        current_segment.append(token_id)

        if len(current_segment) >= max_length:
            segments.append(current_segment)
            current_segment = []

    # Append the remaining tokens if any
    if current_segment:
        segments.append(current_segment)

    return segments


# load final map pickle
logger.info("Loading the final map pickle")
file = open('final_map_year.pkl', 'rb')
final_map = pickle.load(file)
file.close()
logger.info("Loading complete")

sentiment_store = {}
max_length = 510  # Maximum length allowed for each segment
model_name = 'bert-large-uncased'  # Example model name
tokenizer = BertTokenizer.from_pretrained(model_name)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

bert="bert-large-uncased"
binary = False
config = BertConfig.from_pretrained(bert)
if not binary:
    config.num_labels = 5
    
model = BertForSequenceClassification.from_pretrained(bert, config=config)
logger.info("Loading Weigths for torch model from pickle")
model = torch.load("bert-large-uncased__all__fine__e17.pickle", map_location=torch.device('cpu'))
logger.info("Completed Loading Weigths for torch model from pickle")
model = model.to(device)

# final tokenizer
from pytorch_transformers import BertTokenizer
tokenizer_final = BertTokenizer.from_pretrained("bert-large-uncased")

for key, val in final_map.items():
    sentiment_store[key] = []
    for i in tqdm(range(len(val))):
        insert_val = []
        input_sequence = val[i][0]
        file_name = val[i][1]
        year = val[i][2]
        insert_val.append(file_name)
        insert_val.append(input_sequence)
        insert_val.append(year)
        segments = split_input_sequence(input_sequence, max_length, tokenizer)
        # Process each segment separately
        p = 0
        seg_not_done_count = 0
        for segment in segments:
            # Convert the segment back to a string of characters if necessary
            segment_string = tokenizer.decode(segment, skip_special_tokens=True)

            tokens = tokenizer_final.encode(segment_string)
            tokens_tensor = torch.tensor([tokens]).to(device)
            try:
                with torch.no_grad():
                    outputs = model(tokens_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=-1)
                    p += torch.argmax(probs).item()
            except:
                seg_not_done_count += 1
        if len(segments) - seg_not_done_count == 0:
            insert_val.append(-1)
        else:
            p = p/(len(segments)-seg_not_done_count)
            insert_val.append(p)
        insert_val.append(seg_not_done_count)
        
        sentiment_store[key].append(insert_val)
            

with open('sentiment_store.pickle', 'wb') as handle:
    pickle.dump(sentiment_store, handle, protocol=pickle.HIGHEST_PROTOCOL)











