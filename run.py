import argparse
import os
import json
from collections import Counter
from utils import *
from llm_utils import *

COT_PROMPT = "Let's think step by step."
DIRECT_ANS_PROMPT = "The answer is"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference for mixed data")
    parser.add_argument(
        "--demo_selection_method", type=str, default="similarity-based", choices=["diversity-based", "random-based", "similarity-based", "prompting-based"],
        help="method of selecting demos."
    )
    parser.add_argument(
        "--demo_save_dir", type=str, default="./demos_identification", help="demos for scenario identification."
    )
    parser.add_argument(
        "--mixed_data_save_dir", type=str, default="./mixed", help="where to save the mixed data."
    )
    parser.add_argument(
        "--demo_inference_save_dir", type=str, default="./demos_inference", help="demos for inference."
    )
    parser.add_argument(
        "--experimental_result_dir", type=str, default="./experiments_inference", help="where to save experimental results"
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
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument("--random_seed", type=int, default=666, help="random seed")
    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode"
    )
    args = parser.parse_args()
    return args


def scenario_identification(args, input):
    # get demos for identification
    demo_for_identification = f'./{args.demo_save_dir}/{args.input_style}_{args.output_style}_random'
    demos = []
    cached_f = open(demo_for_identification, 'r')
    for line in cached_f.readlines():
        line = json.loads(line)
        demos.append(line)
    demos_string = "\n\n".join(demos)
    demos_string += "\n\n"
    #print(demos_string)

    log_dict = {"1st": "", "2nd": "", "3rd": ""}

    # add input to the demo
    demos_string_input = demos_string + input + "\nType: "
    response = decoder_for_gpt(demos_string_input, engine=args.engine, max_length=32)
    
    pred_type = type_cleansing(args, response)   #string or tuple
    log_dict["1st"] = pred_type
    #print(pred_type)

    if pred_type == "UNDEFINED":
        demos_string_again = demos_string + input + "\nType (choosing from <arithmetic, short-answer>, <arithmetic, multiple-choice>, <commonsense, multiple-choice>, <commonsense, yes-no>, <symbolic, short-answer>, <symbolic, yes-no>): "
        response = decoder_for_gpt(demos_string_again, engine = args.engine, max_length=32)
        pred_type = type_cleansing(args, response)
        log_dict["2nd"] = pred_type
        if pred_type == "UNDEFINED":
            choices = [('arithmetic','short-answer'), ('arithmetic','multiple-choice'), \
                        ('commonsense','multiple-choice'), ('commonsense','yes-no'), \
                        ('symbolic','yes-no'), ('symbolic','short-answer')]
            pred_type = random.sample(choices, 1)[0]
            log_dict["3rd"] = pred_type
    #print(pred_type)
    return pred_type, log_dict


def prompt_constructor(args, pred_type):
    # only for [cat-form] output style, to be updated for other output styles
    cat, form = pred_type
    demo_for_inference = f"{args.demo_inference_save_dir}/{args.demo_selection_method}/{cat}_{form}"
    
    x, z, y = [], [], []
    with open(demo_for_inference, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["demo"]
        for line in json_data:
            x.append(line['question'])
            z.append(line['rationale'])
            y.append(line['pred_ans'])
    index_list = list(range(len(x)))

    demo_text = ""
    for i in index_list:
        demo_text += x[i] + " " + z[i] + " " + \
                    DIRECT_ANS_PROMPT + " " + y[i] + ".\n\n"

    return demo_text


def prompt_constructor_rand_sim(args, pred_type):
    cat, form = pred_type
    demo_for_inference = f"{args.demo_inference_save_dir}/{args.demo_selection_method}/{cat}_{form}"
    all_demos = []

    with open(demo_for_inference) as f:
        for line in f.readlines():
            json_line = json.loads(line)
            x, z, y = [], [], []
            for dem in json_line:
                x.append(dem['question'])
                z.append(dem['rationale'])
                y.append(dem['pred_ans'])
            index_list = list(range(len(x)))
            demo_text = ""
            for i in index_list:
                demo_text += x[i] + " " + z[i] + " " + \
                            DIRECT_ANS_PROMPT + " " + y[i] + ".\n\n"
            all_demos.append(demo_text)
    
    return all_demos


def run_inference(args, resume_id=0, current_type=""):
    # create mixed data for inference
    mixed_file = f"{args.mixed_data_save_dir}/{args.output_style}"
    assert os.path.exists(mixed_file)    # ensure the mixed data preprocessing has been completed
    with open(mixed_file) as f:
        mixed_data = json.load(f)
    # Inference...
    for type, type_data in mixed_data.items():
        if current_type != "" and type != current_type:
            continue
        output_path = f"./{args.experimental_result_dir}/{args.output_style}_{args.demo_selection_method}_{type}"
        corpus, rationales, pred_answers, gold_answers, ori_dataset, gold_type = type_data
        questions = ["Q: "+ cor +"\nA: " for cor in corpus]
        print(f"*************************{type}*************************")
        print(f"Data size: {len(questions)}")
        total = 0
        correct_list = []
        with open(output_path, 'a') as f:
            for i, question in enumerate(questions):
                if i < resume_id -1:
                    continue
                print('*************************')
                print("{}st data".format(i+1))
                input_question = question
                pure_question = corpus[i]
                gold_ans = gold_answers[i]
                ori_ds = ori_dataset[i]
                zero_shot_rationale = rationales[i]
                zero_shot_pred_ans = pred_answers[i]
                
                # get corresponding demos for inference
                pred_type, log_type = scenario_identification(args, pure_question)
                if args.demo_selection_method == "diversity-based":
                    demos_string = prompt_constructor(args, pred_type)
                elif args.demo_selection_method == "random-based" or "similarity-based":
                    all_demos = prompt_constructor_rand_sim(args, pred_type)
                    demos_string = all_demos[i]

                # add demos to the input
                input = demos_string + question + COT_PROMPT
                # generate the rationale
                rationale = decoder_for_gpt(input, engine=args.engine, max_length=256)
                # extract the answer
                pred_ans = answer_cleansing(args, rationale, pred_type)

                print("==============Input Question==============")
                print(input_question)
                print("==============Pred Answer==============")
                print(pred_ans)
                print("==============Gold Answer==============")
                print(gold_ans)
                print("==============Pred Type==============")
                print(f"<{pred_type[0]}, {pred_type[1]}>")
                print("==============Gold Type==============")
                print(gold_type)

                # Checking answer ...
                correct = (np.array([pred_ans]) == np.array([gold_ans])).sum().item()
                correct_list.append(correct)
                total += 1 #np.array([y]).size(0)

                record_data = {
                    'question': pure_question,
                    'pred_ans': pred_ans,
                    'gold_ans': gold_ans,
                    'rationale': rationale,
                    'pred_type': f"<{pred_type[0]}, {pred_type[1]}>",
                    'gold_type': gold_type,
                    'log_type': log_type,
                    'ori_dataset': ori_ds, 
                    'zero_rationale': zero_shot_rationale,
                    'zero_pred_ans': zero_shot_pred_ans
                }

                record = json.dumps(record_data)
                f.write(record + '\n')

                if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                    break
                    
        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))

    return


def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    current_type = "commonsense_multiple-choice"
    resume_id = 0
    run_inference(args, resume_id=resume_id, current_type=current_type)
    

if __name__ == "__main__":
    main()
