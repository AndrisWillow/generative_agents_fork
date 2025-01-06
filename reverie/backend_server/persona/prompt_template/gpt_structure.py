"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import openai
import time 

from utils import *

import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name_or_path = "meta-llama/Llama-3.2-3B-Instructt" # meta-llama/Llama-3.2-3B-Instruct # meta-llama/Meta-Llama-3-8B-Instruct
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=config, device_map="auto")

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

model.eval()

def LLM_call(prompt, max_tokens=50, temperature=0.7):
    print("called Llama 3")
    print(max_tokens, temperature)
    model_inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model_inputs["input_ids"].to(device)
        # Note the length of the input
    input_length = tokens.shape[1]
    generation_output = model.generate(
        tokens,
        max_new_tokens = max_tokens,
        do_sample=True,
        temperature=0.7, # constant for debugging
        pad_token_id=tokenizer.eos_token_id
    )
    new_tokens = generation_output[0, input_length:].tolist()  # Get only the new token ids
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f'\n {output}')
    return output

llm = LLM_call

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt): 
  temp_sleep()
  try:
    print("ChatGPT_single_request")
    response = llm(prompt)
  except Exception as e:
    print(e)
    ### TODO: Add map-reduce or splitter to handle this error.
    return "LLM ERROR"
  return response


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-4", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt,parameters): 
  """
  Given a prompt, make a request to LLM server and returns the response. 
  ARGS:
    prompt: a str prompt 
    parameters: optional
  RETURNS: 
    a str of LLM's response. 
  """
  # temp_sleep()
  try:
    print("ChatGPT_request")
    response = llm(prompt, parameters["max_tokens"], parameters["temperature"])
  except Exception as e:
    print(e)
    ### TODO: Add map-reduce or splitter to handle this error.
    return "LLM ERROR"
  return response

def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt,parameters): 
  """
  Given a prompt, make a request to LLM server and returns the response. 
  ARGS:
    prompt: a str prompt 
    parameters: optional 
  RETURNS: 
    a str of LLM's response. 
  """
  # temp_sleep()
  try:
    response = llm(prompt, parameters["max_tokens"], parameters["temperature"])
  except Exception as e:
    print(e)
    ### TODO: Add map-reduce or splitter to handle this error.
    return "LLM ERROR"
  return response


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response

def get_embedding(text: str):
    """
    Compute a simple embedding by averaging the final hidden states from a decoder-only Llama model.
    """
    # 1. Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 2. Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,  # request all hidden states
            return_dict=True            # return as a dictionary-like object
        )
        # outputs.hidden_states is a tuple of hidden states from all layers
        # The final layer is the last element in that tuple:
        last_hidden_state = outputs.hidden_states[-1]  # shape: [batch_size, seq_length, hidden_dim]

    # 3. Pool the sequence of hidden states into a single vector (for example, mean-pool):
    embedding = last_hidden_state.mean(dim=1)  # shape: [batch_size, hidden_dim]
    embedding = embedding.squeeze(0)           # shape: [hidden_dim] (for a single example)

    return embedding


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)
