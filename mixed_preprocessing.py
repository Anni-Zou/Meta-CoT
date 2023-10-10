import random
import json
import os
import argparse
import glob
import re
from utils import fix_seed,data_reader
from llm_utils import *



# MCQ: Multiple-Choice Question
# SAQ: Short-Answer Question
# Y/N: Yes-No Question
DATA_INFO = {
    "addsub":{"category": "arithmetic", 
                "form": "short-answer",
                "data_file": "./dataset/AddSub/AddSub.json",
                "pred_file": "./log/addsub_zero_shot_cot.log"},
    "aqua":{"category": "arithmetic", 
                "form": "multiple-choice",
                "data_file": "./dataset/AQuA/test.json",
                "pred_file": "./log/aqua_zero_shot_cot.log"},
    "coin-flip":{"category": "symbolic", 
                "form": "yes-no",
                "data_file": "./dataset/coin_flip/coin_flip.json",
                "pred_file": "./log/coin_flip_zero_shot_cot.log"},
    "commonsensqa":{"category": "commonsense", 
                "form": "multiple-choice",
                "data_file": "./dataset/CommonsenseQA/dev_rand_split.jsonl",
                "pred_file": "./log/commonsensqa_zero_shot_cot.log"},
    "gsm8k":{"category": "arithmetic", 
                "form": "short-answer",
                "data_file": "./dataset/grade-school-math/test.jsonl",
                "pred_file": "./log/gsm8k_zero_shot_cot.log"},
    "last-letters":{"category": "symbolic", 
                "form": "short-answer",
                "data_file": "./dataset/last_letters/last_letters.json",
                "pred_file": "./log/last_letters_zero_shot_cot.log"},
    "multiarith":{"category": "arithmetic", 
                "form": "short-answer",
                "data_file": "./dataset/MultiArith/MultiArith.json",
                "pred_file": "./log/multiarith_zero_shot_cot.log"},
    "singleeq":{"category": "arithmetic", 
                "form": "short-answer",
                "data_file": "./dataset/SingleEq/questions.json",
                "pred_file": "./log/singleeq_zero_shot_cot.log"},
    "strategyqa":{"category": "commonsense", 
                "form": "yes-no",
                "data_file": "./dataset/StrategyQA/task.json",
                "pred_file": "./log/strategyqa_zero_shot_cot.log"},
    "svamp":{"category": "arithmetic", 
                "form": "short-answer",
                "data_file": "./dataset/SVAMP/SVAMP.json",
                "pred_file": "./log/svamp_zero_shot_cot.log"},
}



def parse_arguments():
    parser = argparse.ArgumentParser(description="Mixed-Data-Preprocessing")
    parser.add_argument(
        "--load_dir", type=str, default="./data", help="data dir for loaded dataset info"
    )
    parser.add_argument(
        "--input_style", type=str, default="que", choices=["que", "que-ans", "que-cot", "que-cot-ans"],
        help="the input form for identification."
    )
    parser.add_argument(
        "--output_style", type=str, default="cat-form", choices=["task", "category", "form", "cat-form"],
        help="the output form for identification."
    )
    parser.add_argument(
        "--engine", type=str, default="gpt-35-turbo", choices=["gpt-35-turbo", "gpt-35-turbo-16k", "gpt-4", "gpt-4-32k"],
        help="llm engine."
    )
    parser.add_argument(
        "--demo_save_dir", type=str, default="./demos", help="where to save the constructed demonstrations"
    )
    parser.add_argument(
        "--mixed_data_save_dir", type=str, default="./mixed", help="where to save the mixed data"
    )
    parser.add_argument(
        "--data_exp_dir", type=str, default="./data_exp", help="where to save the preliminary experimental results"
    )
    parser.add_argument(
        "--mixed_data_exp_dir", type=str, default="./experiment", help="where to save the preliminary experimental results"
    )
    parser.add_argument(
        "--demo_sampling_method", type=str, default="random", choices=["random", "random-manual", "manual"], help="method to sample demos"
    )
    parser.add_argument("--num_experiment_data", type=int, default=1500, help="total experimental data size")
    parser.add_argument("--run_test", type=bool, default=False, help="whether to conduct the preprocessing preliminary experiments")
    parser.add_argument("--random_seed", type=int, default=192, help="random seed")
    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode"
    )
    args = parser.parse_args()
    return args


def load_data_wcot(args, dataset_name):
    # we find some questions from the pred_file are not complete, so we load questions and answers from the original data file
    '''
    load [question, answer] from the original data file, and [cot] from the zero-shot-cot file
    input: dataset name
    output: dataset info stored in new files
    '''
    data_file = DATA_INFO[dataset_name]["data_file"]
    pred_file = DATA_INFO[dataset_name]["pred_file"]
    category = DATA_INFO[dataset_name]["category"]
    form = DATA_INFO[dataset_name]["form"]
    dataset_path = f'./{args.load_dir}/{dataset_name}_{category}_{form}'

    if not os.path.exists(args.load_dir):
        os.mkdir(args.load_dir)


    assert not os.path.exists(dataset_path)

    
    corpus, question, rationale, pred_ans, gold_ans = [], [], [], [], []

    with open(pred_file, "r", encoding="utf-8") as fp:
        answer_seg = ""
        for line in fp:
            if "Q: " in line:
                c_question = line.strip()
                pure_question = c_question.strip("Q: ")
            if "A: " in line:
                answer_seg = line
            elif "Therefore" in line and "the answer" in line:
                c_rationale = answer_seg

            elif answer_seg != "":
                answer_seg += line
            if "pred_mode" in line:
                c_pred_ans = line.split(":")[1].strip()
            if "GT :" in line:
                c_gold_ans = line.split(":")[1].strip()
                c_rationale = c_rationale.replace("A: Let's think step by step.", "Let's think step by step.")
                c_question = c_question + "\nA: "

                corpus.append(pure_question)
                question.append(c_question)
                rationale.append(c_rationale)
                pred_ans.append(c_pred_ans)
                if args.debug:
                    gold_ans.append(c_gold_ans)
                answer_seg = ""

        questions, _ = data_reader(dataset_name, data_file)

        for i, cor in enumerate(corpus):
            for que in questions:
                if cor in que and cor != que:
                    corpus[i] = que


        dataset_info = {"dataset": dataset_name, "category": category, "form": form,
                "corpus": corpus, "question": question, "rationale": rationale,
                "pred_ans": pred_ans, "gold_ans": gold_ans}


        with open(dataset_path, 'w') as f:
            dataset_json = json.dumps(dataset_info)
            f.write(dataset_json)

    return


def read_loaded_data(data_dir):
    '''
    read loaded data from dir
    input: data_dir (contain solely a dict)
    output: dataset_info (dict)
    '''
    with open(data_dir, 'r') as data_file:
        dataset_info = json.load(data_file)
    return dataset_info


def set_input(args, info_tuple):
    '''
    input: a tuple: (question, rationale, predans, goldans, ori_dataset)
    output: string referring to a specified input style
    '''
    input_style = args.input_style
    question, rationale, predans, goldans, ori_dataset = info_tuple

    if input_style == "que":
        input_data = "Q: "+ question
    elif input_style == "que-ans":
        input_data = "Q: " + question + "\nA: " + predans
    elif input_style == "que-cot":
        input_data = "Q: " + question + "\nA: " + rationale
    elif input_style == "que-cot-ans":
        input_data = "Q: " + question + "\nA: " + rationale + "Therefore, the answer is " + predans
    
    return input_data


def set_output(args, dataset_info):
    '''
    input: dataset_info (dict)
    output: string referring to a specified style
    '''
    output_style = args.output_style

    if output_style == "task":
        task_name = dataset_info["dataset"]
        output_data = task_name
    elif output_style == "category":
        category = dataset_info["category"]
        output_data = category
    elif output_style == "form":
        form = dataset_info["form"]
        output_data = form
    elif output_style == "cat-form":
        category = dataset_info["category"]
        form = dataset_info["form"]
        output_data = f'<{category}, {form}>'
    elif output_style == "task-cat":
        task_name = dataset_info["dataset"]
        category = dataset_info["category"]
        output_data = f'<{task_name}, {category}>'
    assert type(output_data) == str

    return output_data


def combine_inout_data(input_data, output_data):
    '''
    input: input_data (string), output_data (string)
    output: a string combines the in&out data
    '''
    combined =  input_data + "\n" + "Scenario: " + output_data
    return combined


def split_and_merge_data(args):
    '''
    split and merge the data based on the output_style
    output: dict: {'A': [corpus(list), rationale(list), predans(list), goldans(list), ori_dataset(list), outputdata(string)]}
    '''
    output_style = args.output_style
    load_dir = args.load_dir
    mixed_data_save_dir = args.mixed_data_save_dir

    mixed_data_dir = f'{mixed_data_save_dir}/{output_style}'

    if not os.path.exists(mixed_data_save_dir):
        os.mkdir(mixed_data_save_dir)

    if os.path.exists(mixed_data_dir):
        cached_f = open(mixed_data_dir, 'r')
        mixed_data = json.load(cached_f)
        return mixed_data

    if output_style == "task":
        data_dirs = glob.glob(f'{load_dir}/*')

        tasks = {}
        for data_dir in data_dirs:
            task = data_dir.split('/')[-1].split('_')[0]
            if task in tasks.keys():
                tasks[task].append(data_dir)
            else:
                tasks[task] = [data_dir]

        d_data = dict.fromkeys(tasks.keys(),[])

        for task, data_dirs_list in tasks.items():
            task_corpus, task_rationale, task_predans, task_goldans, task_oriname= [], [], [], [], []
            for dir in data_dirs_list:
                dataset_info = read_loaded_data(dir)
                task_corpus += dataset_info["corpus"]
                task_rationale += dataset_info["rationale"]
                task_predans += dataset_info["pred_ans"]
                task_goldans += dataset_info["gold_ans"]
                task_oriname += [dataset_info['dataset'] for _ in range(len(dataset_info["corpus"]))]
            output_data = set_output(args, dataset_info)
            assert len(task_corpus) == len(task_rationale) == len(task_predans) == len(task_goldans) == len(task_oriname)
            zipped_list = list(zip(task_corpus, task_rationale, task_predans, task_goldans, task_oriname))
            random.shuffle(zipped_list)
            task_corpus, task_rationale, task_predans, task_goldans, task_oriname = zip(*zipped_list)
            d_data[task] = [task_corpus, task_rationale, task_predans, task_goldans, task_oriname, output_data]

    elif output_style == "category":
        data_dirs = glob.glob(f"{load_dir}/*")
        
        category = {}
        for data_dir in data_dirs:
            cat = data_dir.split('/')[-1].split('_')[1]
            if cat in category.keys():
                category[cat].append(data_dir)
            else:
                category[cat] = [data_dir]

        d_data = dict.fromkeys(category.keys(), [])

        for cat, data_dirs_list in category.items():
            cat_corpus, cat_rationale, cat_predans, cat_goldans, cat_oriname= [], [], [], [], []
            for dir in data_dirs_list:
                dataset_info = read_loaded_data(dir)
                cat_corpus += dataset_info["corpus"]
                cat_rationale += dataset_info["rationale"]
                cat_predans += dataset_info["pred_ans"]
                cat_goldans += dataset_info["gold_ans"]
                cat_oriname += [dataset_info['dataset'] for _ in range(len(dataset_info["corpus"]))]
            output_data = set_output(args, dataset_info)
            assert len(cat_corpus) == len(cat_rationale) == len(cat_predans) == len(cat_goldans) == len(cat_oriname)
            zipped_list = list(zip(cat_corpus, cat_rationale, cat_predans, cat_goldans, cat_oriname))
            random.shuffle(zipped_list)
            cat_corpus, cat_rationale, cat_predans, cat_goldans, cat_oriname = zip(*zipped_list)
            d_data[cat] = [cat_corpus, cat_rationale, cat_predans, cat_goldans, cat_oriname, output_data]

    elif output_style == "cat-form":
        data_dirs = glob.glob(f"{load_dir}/*")
        
        cat_forms = {}
        for data_dir in data_dirs:
            cat = data_dir.split('/')[-1].split('_')[1]
            form = data_dir.split('/')[-1].split('_')[-1]
            cat_form = f'{cat}_{form}'
            if cat_form in cat_forms.keys():
                cat_forms[cat_form].append(data_dir)
            else:
                cat_forms[cat_form] = [data_dir]

        d_data = dict.fromkeys(cat_forms.keys(), [])

        for cat_form, data_dirs_list in cat_forms.items():
            catform_corpus, catform_rationale, catform_predans, catform_goldans, catform_oriname= [], [], [], [], []
            for dir in data_dirs_list:
                dataset_info = read_loaded_data(dir)
                catform_corpus += dataset_info["corpus"]
                catform_rationale += dataset_info["rationale"]
                catform_predans += dataset_info["pred_ans"]
                catform_goldans += dataset_info["gold_ans"]
                catform_oriname += [dataset_info['dataset'] for _ in range(len(dataset_info["corpus"]))]
            output_data = set_output(args, dataset_info)
            assert len(catform_corpus) == len(catform_rationale) == len(catform_predans) == len(catform_goldans) == len(catform_oriname)
            zipped_list = list(zip(catform_corpus, catform_rationale, catform_predans, catform_goldans, catform_oriname))
            random.shuffle(zipped_list)
            catform_corpus, catform_rationale, catform_predans, catform_goldans, catform_oriname = zip(*zipped_list)
            d_data[cat_form] = [catform_corpus, catform_rationale, catform_predans, catform_goldans, catform_oriname, output_data]

    elif output_style == "form":
        data_dirs = glob.glob(f"{load_dir}/*")
        
        forms = {}
        for data_dir in data_dirs:
            form = data_dir.split('/')[-1].split('_')[-1]
            if form in forms.keys():
                forms[form].append(data_dir)
            else:
                forms[form] = [data_dir]

        d_data = dict.fromkeys(forms.keys(), [])

        for form, data_dirs_list in forms.items():
            form_corpus, form_rationale, form_predans, form_goldans, form_oriname= [], [], [], [], []
            for dir in data_dirs_list:
                dataset_info = read_loaded_data(dir)
                form_corpus += dataset_info["corpus"]
                form_rationale += dataset_info["rationale"]
                form_predans += dataset_info["pred_ans"]
                form_goldans += dataset_info["gold_ans"]
                form_oriname += [dataset_info['dataset'] for _ in range(len(dataset_info["corpus"]))]
            output_data = set_output(args, dataset_info)
            assert len(form_corpus) == len(form_rationale) == len(form_predans) == len(form_goldans) == len(form_oriname)
            zipped_list = list(zip(form_corpus, form_rationale, form_predans, form_goldans, form_oriname))
            random.shuffle(zipped_list)
            form_corpus, form_rationale, form_predans, form_goldans, form_oriname = zip(*zipped_list)
            d_data[form] = [form_corpus, form_rationale, form_predans, form_goldans, form_oriname, output_data]

    with open(mixed_data_dir, 'w') as f:
        data_json = json.dumps(d_data)
        f.write(data_json + "\n")

    return d_data


def num_sampling(args):
    split_style = args.output_style
    if split_style == 'task' or 'cat-form':
        num_sample = 1
    elif split_style == 'category':
        num_sample = 2
    elif split_style == 'form':
        num_sample = 2
    return num_sample


def get_demo(args, split_data):
    demo_save_dir = args.demo_save_dir  #./demos
    demo_sampling_method = args.demo_sampling_method
    split_style = args.output_style
    input_style = args.input_style
    num_sample = num_sampling(args)

    demo_save_file = f'{demo_save_dir}/{input_style}_{split_style}_{demo_sampling_method}'
    if not os.path.exists(demo_save_dir):
        os.mkdir(demo_save_dir)

    demos = []
    if os.path.exists(demo_save_file):
        cached_f = open(demo_save_file, 'r')
        for line in cached_f.readlines():
            line = json.loads(line)
            demos.append(line)
    else:
        for key, value in split_data.items():
            corpus, rationale, pred_ans, gold_ans, ori_dataset, outputdata = value
            zipped_list = list(zip(corpus, rationale, pred_ans, gold_ans, ori_dataset))
            sampled_demos = random.sample(zipped_list, num_sample)
            for sdemo in sampled_demos:
                demo_string = combine_inout_data(set_input(args, sdemo), outputdata)
            demos.append(demo_string)    

        with open(demo_save_file, 'w') as f:
            for demo in demos:
                demo_json = json.dumps(demo)
                f.write(demo_json + "\n")

    return demos

def get_experiment_data(args, demos, split_data):
    split_style = args.output_style
    input_style = args.input_style
    num_sample_exp = args.num_experiment_data / len(split_data.keys())
    
    demos_comb = [data[-1] for data in demos]
    mixed_data_save_file = f'./{args.data_exp_dir}/{input_style}_{split_style}'
    if not os.path.exists(args.data_exp_dir):
        os.mkdir(args.data_exp_dir)

    mixed_data = []
    
    if os.path.exists(mixed_data_save_file):
        cached_f = open(mixed_data_save_file, 'r')
        for line in cached_f.readlines():
            line = json.loads(line)
            mixed_data.append(line)
    else:
        for key, value in split_data.items():
            corpus, rationale, pred_ans, gold_ans, ori_dataset, outputdata = value
            zipped_list = list(zip(corpus, rationale, pred_ans, gold_ans, ori_dataset))
            num_total = 0
            for i, data in enumerate(zipped_list):
                if num_total < num_sample_exp:
                    if data[-1] in demos_comb:
                        continue
                    else:
                        in_d = set_input(args, data)
                        comb_d = combine_inout_data(in_d, outputdata)
                        cor, rat, pred, gold, ori_d = data
                        mixed_data.append({'in': in_d, 'out': outputdata, 'comb': comb_d,
                                            'question': cor, 'rationale': rat, 'pred_ans': pred, 'gold_ans': gold, 'ori_dataset': ori_d})
                        num_total += 1
                else:
                    break

        with open(mixed_data_save_file, 'w') as f:
            for m_data in mixed_data:
                data_json = json.dumps(m_data)
                f.write(data_json + "\n")

    return mixed_data
        
def type_cleansing(args, type):
    if args.output_style == 'task':
        new_type = re.findall(r'addsub|aqua|gsm8k|multiarith|singleeq|svamp|commonsensqa|strategyqa|coin-flip|last-letters', type)
        if len(new_type) != 0:
            type = new_type[0]
    elif args.output_style == 'category':
        new_cat = re.findall(r'arithmetic|commonsense|symbolic', type)
        if len(new_cat) != 0:
            type = new_cat[0]
    elif args.output_style == 'form':
        new_form = re.findall(r'multiple-choice|short-answer|yes-no', type)
        if len(new_form) != 0:
            type = new_form[0]
    elif args.output_style == 'cat-form':
        new_cat = re.findall(r'arithmetic|commonsense|symbolic', type)
        new_form = re.findall(r'multiple-choice|short-answer|yes-no', type)
        if len(new_cat) != 0 and len(new_form) != 0:
            new_cat = new_cat[0]
            new_form = new_form[0]
            type = f"<{new_cat}, {new_form}>"

    return type
        


def type_identification(args, demo, input):
    demo += input + "\nScenario "

    #if args.output_style == 'task':
    #    demo += "(choosing from 'addsub', 'aqua', 'gsm8k', 'multiarith', 'singleeq', 'svamp', 'commonsensqa', 'strategyqa', 'coin-flip', 'last-letters')"
    #elif args.output_style == 'category':
    #    demo += "(choosing from 'arithmetic', 'commonsense', 'symbolic')"
    #elif args.output_style == 'form':
    #    demo += "(choosing from 'multiple-choice', 'short-answer', 'yes-no')"
    #elif args.output_style == 'cat-form':
    #    demo += "(choosing from '<arithmetic, short-answer>', '<arithmetic, multiple-choice>', '<commonsense, multiple-choice>', '<commonsense, yes-no>', '<symbolic, short-answer>', '<symbolic, yes-no>')"

    demo += ": "
    response  = decoder_for_gpt(demo, engine=args.engine, max_length=32)
    response = response.strip().lower()
    return response




def main(resume_id=0):
    args = parse_arguments()
    fix_seed(args.random_seed)

    
    # load and store dataset info
    for dataset_name in DATA_INFO.keys():
        category = DATA_INFO[dataset_name]["category"]
        form = DATA_INFO[dataset_name]["form"]
        dataset_path = f'./{args.load_dir}/{dataset_name}_{category}_{form}'
        if os.path.exists(dataset_path):
            pass
        else:
            load_data_wcot(args, dataset_name)

    # split and merge data
    split_data = split_and_merge_data(args)

    # get demos based on the output_style
    demos = get_demo(args, split_data)
    demos_string = "\n\n".join(demos)
    demos_string += "\n\n"
    print(demos_string)

    # get mixed experimental data
    mixed_exp_data = get_experiment_data(args, demos, split_data)
    print(len(mixed_exp_data))
    print(mixed_exp_data[0])
    
    # experiment
    if args.run_test:

        output_path = f"./{args.mixed_data_exp_dir}/{args.input_style}_{args.output_style}_{args.engine}_{args.demo_sampling_method}"
        with open(output_path, 'a') as f:
            for i, data in enumerate(mixed_exp_data):
                if i < resume_id -1:
                    continue
                print('*************************')
                print("{}st data".format(i+1))

                input_data = data['in']
                gold_typestr = data['out']
                ori_dataset = data['ori_dataset']
                pred_typestr = type_identification(args, demos_string, input_data)
                pred_typestr = type_cleansing(args, pred_typestr)

                new_data = {'input': input_data, 'gold_type': gold_typestr, 'pred_type': pred_typestr, 'ori_dataset': ori_dataset}

                print("------Input------")
                print(input_data)
                print("------Gold_Ans------")
                print(gold_typestr)
                print("------Pred_Ans------")
                print(pred_typestr)

                record = json.dumps(new_data)
                f.write(record + '\n')
    

    return




if __name__ == "__main__":
    #resume_id = 497
    main()
