from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

texts = ["Hello, my dog is cute", "BERT is an amazing model"]

encoded_dict = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
input_ids = encoded_dict['input_ids']
attention_masks = encoded_dict['attention_mask']

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)

last_hidden_states = outputs.last_hidden_state

print("Last hidden states shape:", last_hidden_states.shape)

first_text_last_hidden_states = last_hidden_states[0, 0, :]
print("First text's [CLS] token's last hidden state:", first_text_last_hidden_states)
