from transformers import AlbertTokenizer, AlbertForQuestionAnswering
import torch

model_name = 'ktrapeznikov/albert-xlarge-v2-squad-v2'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForQuestionAnswering.from_pretrained(model_name)
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors='pt')
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
print(start_scores)
print(end_scores)