# from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# model = GPT2Model.from_pretrained('gpt2-xl')
# text = "say hi."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# tokenizer.decode(output.last_hidden_state)


from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

text = "say hello."  # 또는 "Hello, my name is" 같은 자연스러운 프롬프트
encoded_input = tokenizer(text, return_tensors='pt')

# Generate text
output = model.generate(
    encoded_input['input_ids'],
    max_length=200,
    num_return_sequences=1,
    no_repeat_ngram_size=2
)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)