from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"



# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'When will outage happen?',
    'context': 'The Bangalore Electricity Supply Company Limited has said that several areas in Bengaluru will face power outages today (December 30) and tomorrow (December 31) due to maintenance and other works. The power cuts will take place from around 9 am to 5.30 pm. LIC Colony, Yeshwanthpur, Amruthahalli, TR Nagar,/and RBI Layout are among the areas that will be affected.'
}
print("getting answers")
res = nlp(QA_input)
print(res)
