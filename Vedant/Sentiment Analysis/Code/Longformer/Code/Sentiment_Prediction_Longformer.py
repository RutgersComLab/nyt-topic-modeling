import pickle
import torch
import torch.nn as nn
from loguru import logger
import os
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from tqdm import tqdm

def split_input_sequence(input_sequence, max_length, tokenizer):
    encoded_inputs = tokenizer.encode_plus(
        input_sequence,
        add_special_tokens=False,
        return_tensors='pt'
    )

    input_ids = encoded_inputs['input_ids'][0]

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
max_length = 4094  # Maximum length allowed for each segment
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

binary = False
if not binary:
    num_labels = 5
else:
    num_labels = 2

model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels = num_labels)

logger.info("Loading Weigths for torch model from pickle")
model = torch.load("longformer__all__fine__e4.pickle", map_location=torch.device('cpu'))
logger.info("Completed Loading Weigths for torch model from pickle")
model = model.to(device)

tokenizer_final = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

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


