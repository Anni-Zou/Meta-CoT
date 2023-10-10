'''
Modified from https://github.com/amazon-science/auto-cot
'''
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Demo Creation for Inference")
    parser.add_argument(
        "--mixed_data_save_dir", type=str, default="./mixed", help="where to save the mixed data"
    )
    parser.add_argument(
        "--demo_inference_save_dir", type=str, default="./demos_inference", help="where to save the demo for inference"
    )
    parser.add_argument(
        "--demo_sampling_method", type=str, default="diversity", choices=["random", "diversity", "similarity", "prompting"], help="method to sample demos"
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument(
        "--output_style", type=str, default="cat-form", choices=["task", "category", "form", "cat-form"],
        help="the output form for identification."
    )
    parser.add_argument("--random_seed", type=int, default=399, help="random seed")
    parser.add_argument("--num_clusters", type=int, default=8, help="number of clusters")
    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode"
    )
    args = parser.parse_args()
    return args


def sampling_parameter(args, type):
    num_clusters = args.num_clusters
    max_ra_len = args.max_ra_len
    if type == "arithmetic_multiple-choice":
        num_clusters = 4
    elif type == "commonsense_multiple-choice":
        num_clusters = 7
    elif type == "commonsense_yes-no":
        num_clusters = 6
    elif type == "symbolic_short-answer":
        num_clusters = 4
        max_ra_len = 7
    return num_clusters, max_ra_len


def read_mixed_data(args):
    data_dir = f"{args.mixed_data_save_dir}/{args.output_style}"
    with open(data_dir, 'r') as f:
        mixed_data = json.load(f)
    print("Successfully loading mixed data!")
    return mixed_data


def randomly_based(args, mixed_data):
    for type, type_data in mixed_data.items():
        print(f"Begin demo construction for {type}")
        type_demo_save_dir = f"{args.demo_inference_save_dir}/{args.demo_sampling_method}-based/{type}"
        num_clusters, max_ra_len = sampling_parameter(args, type)
        questions, rationales, predans, goldans, ori_dataset, _ = type_data
        corpus = ["Q: " + que + "\nA:" for que in questions]
        index_list = [i for i in range(len(corpus))]
        all_demos = []
        for i, element in enumerate(corpus):
            demos = []
            sampled_index = random.sample(index_list, len(corpus))
            while len(demos) < num_clusters:
                for index in sampled_index:
                    if len(demos) >= num_clusters:
                        break
                    if index == i:
                        continue
                    c_rationale = rationales[index].strip()
                    c_pred_ans = predans[index].strip()
                    if len(corpus[index].strip().split()) <= 60 \
                        and len(c_rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and c_rationale[-1] == "." and c_pred_ans != "":
                        if "arithmetic" in type and "multiple-choice" in type:
                            if not (c_pred_ans.strip() in c_rationale.split(".")[-2] or c_pred_ans.strip() in c_rationale.split()[-10:]):
                                continue
                        c_question = corpus[index]
                        c_rationale = c_rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                        c_rationale = " ".join(c_rationale.split())
                        c_oridataset = ori_dataset[index]
                        if args.debug:
                            c_gold_ans = goldans[index]
                        else:
                            c_gold_ans = None
                        demo_element = {
                            "question": c_question,
                            "rationale": c_rationale,
                            "pred_ans": c_pred_ans,
                            "gold_ans": c_gold_ans,
                            "ori_dataset": c_oridataset
                        }
                        demos.append(demo_element)
            all_demos.append(demos)
        
        with open(type_demo_save_dir, 'w') as write_f:
            for demos in all_demos:
                demo_json = json.dumps(demos)
                write_f.write(demo_json + "\n")
  
    return


def similarity_based(args, mixed_data):
    for type, type_data in mixed_data.items():
        similarity_demo_type(args, type, type_data)
    return


def similarity_demo_type(args, type, type_data):
    encoder = SentenceTransformer(args.encoder)
    type_demo_save_dir = f"{args.demo_inference_save_dir}/{args.demo_sampling_method}-based/{type}"
    num_clusters, max_ra_len = sampling_parameter(args, type)
    questions, rationales, predans, goldans, ori_dataset, _ = type_data
    corpus = ["Q: " + que + "\nA: " for que in questions]
    corpus_embeddings = encoder.encode(corpus)

    arg_matrix = retrieve_top_similarity(corpus_embeddings)

    all_demos = []
    for i, element in enumerate(corpus):
        demos = []
        ranged_index = arg_matrix[i]
        while len(demos) < num_clusters:
            for index in ranged_index:
                if len(demos) >= num_clusters:
                    break
                c_rationale = rationales[index].strip()
                c_pred_ans = predans[index].strip()
                if len(corpus[index].strip().split()) <= 60 \
                    and len(c_rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and c_rationale[-1] == "." and c_pred_ans != "":
                    if "arithmetic" in type and "multiple-choice" in type:
                        if not (c_pred_ans.strip() in c_rationale.split(".")[-2] or c_pred_ans.strip() in c_rationale.split()[-10:]):
                            continue
                    c_question = corpus[index]
                    c_rationale = c_rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                    c_rationale = " ".join(c_rationale.split())
                    c_oridataset = ori_dataset[index]
                    if args.debug:
                        c_gold_ans = goldans[index]
                    else:
                        c_gold_ans = None
                    demo_element = {
                        "question": c_question,
                        "rationale": c_rationale,
                        "pred_ans": c_pred_ans,
                        "gold_ans": c_gold_ans,
                        "ori_dataset": c_oridataset
                    }
                    demos.append(demo_element)
        all_demos.append(demos)
    with open(type_demo_save_dir, 'w') as write_f:
        for demos in all_demos:
            demo_json = json.dumps(demos)
            write_f.write(demo_json + "\n")

    return


def diversity_based(args, mixed_data):
    for type, type_data in mixed_data.items():
        diversity_demo_type(args, type, type_data)
    return


def diversity_demo_type(args, type, type_data):
    encoder = SentenceTransformer(args.encoder)
    type_demo_save_dir = f"{args.demo_inference_save_dir}/{args.demo_sampling_method}-based/{type}"
    questions, rationales, predans, goldans, ori_dataset, _ = type_data
    corpus = ["Q: " + que + "\nA:" for que in questions]
    corpus_embeddings = encoder.encode(corpus)

    # Perform kmean clustering
    print(f"Begin clustering for {type}")
    num_clusters, max_ra_len = sampling_parameter(args, type)
    print(num_clusters)
    print(max_ra_len)
    clustering_model = KMeans(n_clusters=num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for _ in range(num_clusters)]

    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for _ in range(num_clusters)]
    clustered_idx = [[] for _ in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    demos = []

    for i in range(len(clustered_dists)):
        print("Cluster ", i+1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)

        for element in top_min_dist:
            min_idx = element[0]
            c_rationale = rationales[clustered_idx[i][min_idx]].strip()
            c_pred_ans = predans[clustered_idx[i][min_idx]].strip()

            if len(corpus[clustered_idx[i][min_idx]].strip().split()) <= 60 \
                and len(c_rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and c_rationale[-1] == "." and c_pred_ans != "":
                if 'arithmetic' in type and 'multiple-choice' in type:
                    if not (c_pred_ans.strip() in c_rationale.split(".")[-2] or c_pred_ans.strip() in c_rationale.split()[-10:]):
                        continue
                c_question = corpus[clustered_idx[i][min_idx]]
                c_rationale = c_rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                c_rationale = " ".join(c_rationale.split())
                c_oridataset = ori_dataset[clustered_idx[i][min_idx]]
                if args.debug:
                    c_gold_ans = goldans[clustered_idx[i][min_idx]]
                else:
                    c_gold_ans = None
                demo_element = {
                    "question": c_question,
                    "rationale": c_rationale,
                    "pred_ans": c_pred_ans,
                    "gold_ans": c_gold_ans,
                    "ori_dataset": c_oridataset
                }
                demos.append(demo_element)
                print(c_question)
                print(c_rationale)
                print(c_pred_ans)
                print(c_gold_ans)
                print(c_oridataset)
                print("")
                break
    demos = {"demo": demos}

    with open(type_demo_save_dir, 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    return


def create_demos_inference(args, mixed_data):
    demo_sampling_method = args.demo_sampling_method
    demo_folder = f"{args.demo_inference_save_dir}/{args.demo_sampling_method}-based"
    if not os.path.exists(demo_folder):
        os.makedirs(demo_folder)

    if demo_sampling_method == "diversity":
        diversity_based(args, mixed_data)
    elif demo_sampling_method == "random":
        randomly_based(args, mixed_data)
    elif demo_sampling_method == "similarity":
        similarity_based(args, mixed_data)

    return 


def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    mixed_data = read_mixed_data(args)
    create_demos_inference(args, mixed_data)


if __name__ == "__main__":
    main()
