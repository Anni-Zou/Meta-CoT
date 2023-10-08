import re
import random
import torch
import json
import numpy as np
from numpy.linalg import norm

# set the random seed for reproducibility
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def data_reader(dataset, dataset_path):
    
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if dataset == "aqua":
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "(" + "(".join(json_res["options"])
          choice = choice.replace("(", " (").replace(")", ") ")
          choice = "Answer Choices:" + choice
          questions.append(json_res["question"].strip() + " " + choice)
          answers.append(json_res["correct"])
  
    elif dataset == "gsm8k":
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          questions.append(json_res["question"].strip())
          answers.append(json_res["answer"].split("#### ")[-1])
  
    elif dataset == "commonsensqa":
      with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices:"
          for c in json_res["question"]["choices"]:
              choice += " ("
              choice += c["label"]
              choice += ") "
              choice += c["text"]
          questions.append(json_res["question"]["stem"].strip() + " " + choice)
          answers.append(json_res["answerKey"])

    elif dataset in ("addsub", "multiarith", "singleeq"):
      with open(dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
          q = line["sQuestion"].strip()
          a = str(line["lSolutions"][0])
          if a[-2:] == ".0":
              a = a[:-2]
          questions.append(q)
          answers.append(a)
        
    elif dataset == "strategyqa":
      with open(dataset_path) as f:
        json_data = json.load(f)["examples"]
        for line in json_data:
          q = line["input"].strip()
          a = int(line["target_scores"]["Yes"])
          if a == 1:
              a = "yes"
          else:
              a = "no"
          questions.append(q)
          answers.append(a)
        
    elif dataset == "svamp":
      with open(dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)
            
          
    elif dataset in ("coin-flip", "last-letters"):
      with open(dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    #q_len_list = []
    #for q in questions:
    #    q_len_list.append(len(q.split(" ")))
    
    #print("dataset : {}".format(args.dataset))
    #print("data size : {}".format(len(answers)))
    
    return questions, answers



def answer_extraction_prompt(args, pred_type):
    extr_prompt = '\nTherefore, the answer is'  #default answer extraction prompt (if pred_type is UNDEFINED)
    if args.output_style == "cat-form":
        if 'multiple-choice' in pred_type:
            extr_prompt = '\nTherefore, among A through E, the answer is'
        elif 'yes-no' in pred_type:
            extr_prompt = '\nTherefore, the answer (Yes or No) is'
        else:
            if 'arithmetic' in pred_type:
                extr_prompt = '\nTherefore, the answer (arabic numerals) is'
            elif 'symbolic' in pred_type:
                extr_prompt = '\nTherefore, the answer is'
    elif args.output_style == "task":
        if pred_type in ('multiarith', 'gsm8k', 'addsub', 'singleeq', 'svamp'):
            extr_prompt = '\nTherefore, the answer (arabic numerals) is'
        elif pred_type in ('aqua', 'commonsensqa'):
            extr_prompt = '\nTherefore, among A through E, the answer is'
        elif pred_type in ('strategyqa', 'coin-flip'):
            extr_prompt = '\nTherefore, the answer (Yes or No) is'
        else:
            extr_prompt = '\nTherefore, the answer is'
    elif args.output_style == "form":
        if pred_type == 'multiple-choice':
            extr_prompt = '\nTherefore, among A through E, the answer is'
        elif pred_type == 'yes-no':
            extr_prompt = '\nTherefore, the answer (Yes or No) is'
        else:
            extr_prompt = '\nTherefore, the answer is'  # we cannot distinguish whether a saq question is arithmetical type or not
    # we set the answer extraction prompt as default for [category] output style
    return extr_prompt


def type_cleansing(args, type):
    new_type = type.strip().lower()
    type = "UNDEFINED"
    if args.output_style == 'task':
        new_type = re.findall(r'addsub|aqua|gsm8k|multiarith|singleeq|svamp|commonsensqa|strategyqa|coin-flip|last-letters', new_type)
        if len(new_type) != 0:
            type = new_type[0]
    elif args.output_style == 'category':
        new_cat = re.findall(r'arithmetic|commonsense|symbolic', new_type)
        if len(new_cat) != 0:
            type = new_cat[0]
    elif args.output_style == 'form':
        new_form = re.findall(r'multiple-choice|short-answer|yes-no', new_type)
        if len(new_form) != 0:
            type = new_form[0]
    elif args.output_style == 'cat-form':
        new_cat = re.findall(r'arithmetic|commonsense|symbolic', new_type)
        new_form = re.findall(r'multiple-choice|short-answer|yes-no', new_type)
        if len(new_cat) != 0 and len(new_form) != 0:
            new_cat = new_cat[0]
            new_form = new_form[0]
            if (new_cat, new_form) in {('arithmetic', 'short-answer'), ('arithmetic', 'multiple-choice'), \
                                        ('commonsense', 'multiple-choice'), ('commonsense', 'yes-no'), \
                                            ('symbolic', 'yes-no'), ('symbolic', 'short-answer')}:
                type = (new_cat, new_form)
    return type


def entity_cleansing(ent):
    ent = re.sub("\n|\s*-\s*|\.", ",", ent)
    ent = ent.split(",")
    ent_ = []
    for e in ent:
        e = e.strip()
        if e != "" and e not in ('A', 'B', 'C', 'D', 'E'):
            ent_.append(e)
    return ent_

def knowledge_cleansing(knowledge):
    #print("Knowledge Before: " + knowledge)
    knowledge = knowledge.strip()
    if knowledge.startswith("No, "):
        knowledge = re.sub("No, ", "", knowledge)
    knowledge = re.sub("\s"," ", knowledge)
    #print("Knowledge After: " + knowledge)
    return knowledge

def answer_cleansing(args, ans, pred_type):
    #to be updated for [category]&[form] output style
    direct_answer_trigger_for_fewshot = "The answer is"
    # answer extraction for few-shot/auto-cot pattern 
    ans = ans.split(direct_answer_trigger_for_fewshot)
    answer_flag = True if len(ans) > 1 else False
    ans = ans[-1]

    if args.output_style == 'cat-form':
        if 'multiple-choice' in pred_type:
            ans = re.findall(r'A|B|C|D|E', ans)
        elif 'yes-no' in pred_type:
            ans = ans.lower()
            ans = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", ans)
            ans = ans.split(" ")
            ans = [i for i in ans if i in ("yes", "no")]
        elif 'short-answer' in pred_type:
            if 'arithmetic' in pred_type:
                ans = ans.replace(",", "")
                ans = [s for s in re.findall(r'-?\d+\.?\d*', ans)]
            elif 'symbolic' in pred_type:
                ans = re.sub("\"|\'|\n|\.|\s", "", ans)
                ans = [ans]
    elif args.output_style == 'task':
        if pred_type in ("aqua", "commonsensqa"):
            ans = re.findall(r'A|B|C|D|E', ans)
        elif pred_type in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
            ans = ans.replace(",", "")
            ans = [s for s in re.findall(r'-?\d+\.?\d*', ans)]
        elif pred_type in ("strategyqa", "coin-flip"):
            ans = ans.lower()
            ans = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", ans)
            ans = ans.split(" ")
            ans = [i for i in ans if i in ("yes", "no")]
        elif pred_type == "last-letters":
            ans = re.sub("\"|\'|\n|\.|\s", "", ans)
            ans = [ans]
    
    # If there is no candidate in list, null is set.
    if len(ans) == 0:
        ans = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            ans = ans[0]
        else:
            # choose the last element in list ...
            ans = ans[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if ans != "":
        if ans[-1] == ".":
            ans = ans[:-1]

    return ans


def cosine_similarity(vecA, vecB):
    '''
    Input: two single vectors with the same shape: (dim, )
    Output: cosine similarity of the two single vectors: float
    '''
    return np.dot(vecA, vecB)/(norm(vecA)*norm(vecB))

def cosine_similarity_matrix(vectors):
    '''
    Input: corpus embeddings: (num_corpus, dim)
    Output: similarity matrix: (num_corpus, num_corpus)
    '''
    return np.array([[cosine_similarity(vec1, vec2) for vec2 in vectors] for vec1 in vectors])

def retrieve_top_similarity(vectors):
    '''
    Input: corpus embeedings: (num_corpus, dim) & top-k to be retrieved
    Output: arg_matrix : (num_corpus, k) referring to the index in the corpus
    '''
    sim_matrix = cosine_similarity_matrix(vectors)
    arg_matrix = np.array([np.argsort(cor)[::-1][1:] for cor in sim_matrix])
    return arg_matrix

def answer_cleansing_assumed(args, ans, pred_type):
    #to be updated for [category]&[form] output style
    direct_answer_trigger_for_fewshot = "The answer is"
    # answer extraction for few-shot/auto-cot pattern 
    ans = ans.split(direct_answer_trigger_for_fewshot)
    answer_flag = True if len(ans) > 1 else False
    ans = ans[-1]

    answer_list = []

    ans1 = re.findall(r'A|B|C|D|E', ans)
    answer_list.append(ans1)

    ans2 = ans.lower()
    ans2 = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", ans2)
    ans2 = ans2.split(" ")
    ans2 = [i for i in ans2 if i in ("yes", "no")]
    answer_list.append(ans2)

    ans3 = ans.replace(",", "")
    ans3 = [s for s in re.findall(r'-?\d+\.?\d*', ans3)]
    answer_list.append(ans3)

    ans4 = re.sub("\"|\'|\n|\.|\s", "", ans)
    ans4 = [ans4]
    answer_list.append(ans4)

    new_ans_list = []
    for ans in answer_list:
        if len(ans) == 0:
            ans = ""
        else:
            if answer_flag:
                # choose the first element in list ...
                ans = ans[0]
            else:
                # choose the last element in list ...
                ans = ans[-1]
        if ans != "":
            if ans[-1] == ".":
                ans = ans[:-1]
        if ans != "":
            new_ans_list.append(ans)
    
    if len(new_ans_list) == 0:
        res = ""
    elif len(new_ans_list) == 1:
        res = new_ans_list[0]
    else:
        res = random.sample(new_ans_list, 1)[0]

    return res
