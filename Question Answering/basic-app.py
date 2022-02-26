# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

from QA import QA
 
# Declaring our FastAPI instance
app = FastAPI()

# initializing the model
global model 
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
global tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def question_answer(question, text):
    input_ids = tokenizer.encode(question, text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_idx+1
    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
                
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
    return answer.capitalize()

 
# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Hello, World!'}
 
# Defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name : str):
    # Defining a function that takes only string as input and output the
    # following message.
    return {'message': f'Hello, {name}'}


# Defining path for posting data
@app.post('/answer')
def fetch_answer(data:QA):
    data = data.dict()
    context = data['context']
    question = data['question']
    answer = question_answer(context, question)
    return {
        'answer': answer
    }