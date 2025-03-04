from montecarlo.node import Node
from montecarlo.montecarlo import MonteCarlo

from sql import score_func
from cmdline import args
from default_prompts import problems
from common import limit_depth, count_depth
from common_stats import stats

import llm
import json, copy, re, os, sys, ipdb
from utils import check_consistency, make_duplication

# export CUDA_VISIBLE_DEVICES=0 && nohup python -u run.py >> test.log &

os.environ["TOKENIZERS_PARALLELISM"] = "false"

predicted_sqls = []
def generate_complete(state, true_sql, db_id, is_leaf, depth):
    if len(state) != 0 and state[-1] == ';':
        score = -2 
        sql_text = None
        return score, sql_text
    
    beam_search_num = args.beam_search_num
    if args.infer_methd == 'vllm':
        texts = llm.generate_by_vllm(state, beam_search_num)
    else:
        texts = llm.generate(state, beam_search_num)
    sql_texts_uniq = list()
    for text in texts:        
        sql_text = text
        json_pattern = r"```json\n(.*?)\n```"
        if re.search(json_pattern, sql_text):
            sql_text = re.findall(json_pattern, sql_text)[0]

        if is_leaf:
            sql_pattern = r"```sql\n(.*?)\n```"
            sql_match = re.search(sql_pattern, sql_text, re.DOTALL)
            if sql_match:
                sql_text = next((group for group in sql_match.groups() if group is not None), None)
                sql_text = sql_text.strip().replace('\n', ' ').strip()
                sql_text = sql_text.replace("```", '').strip()
            else:
                try:
                    sql_pattern = r"\{.*?\}"
                    sql_match = re.search(sql_pattern, sql_text, re.DOTALL)
                    if sql_match:
                        sql_dict = sql_match.group()
                        sql_dict = eval(sql_dict)
                        sql_text = next(iter(sql_dict.values()))
                        if type(sql_text) is list:
                            if type(sql_text[0]) is str:
                                sql_text = sql_text[0]  
                except:
                    pass
        else:
            sql_text=sql_text.replace('\n', '')
            if depth == 2:
                pattern = r"(\{'tables.*?\})"
            elif depth == 3:
                pattern = r"(\{'columns.*?\})"
            elif depth == 4:
                pattern = r"(\{'conditions.*?\})"
            match = re.search(pattern, sql_text, re.DOTALL)
            if match:
                sql_text = match.group()

        if type(sql_text) is str:
            sql_texts_uniq.append(sql_text)

    texts_uniq = list(set(sql_texts_uniq))
    return texts_uniq

def child_finder(node, true_sql, db_id, schema_items, montecarlo):
    depth = count_depth(node)
    if limit_depth(node):
        return
    true_sql = true_sql + ';'
    child_texts = list()
    prompt = montecarlo.prompt
    step_prompt = montecarlo.step_prompts[depth]
    prompt = node.state + '\n' + step_prompt
    is_leaf_next = ((depth+1) == len(step_prompts))
    texts = generate_complete(prompt, true_sql, db_id, is_leaf_next, depth+1)

    scores = dict()
    if is_leaf_next:
        texts_ =  copy.deepcopy(texts)
        for sql_text in texts_:
            is_consistency = check_consistency(prompt, sql_text, match_level=depth)
            if len(sql_text) > 200:
                texts.remove(sql_text)
                continue
            else:
                score = score_func(sql_text, true_sql, db_id)
            if score < 0:
                score = 0
            if (not is_consistency) and score == 1:
                node.update_win_value(0)
                texts.remove(sql_text)
                continue
            if score == 1:
                montecarlo.solutions.add(sql_text)
            elif score == 0:
                montecarlo.negatives.add(sql_text)
            scores[sql_text] = score

    child_texts = list(set(texts))
    child_texts = make_duplication(child_texts, match_level=depth+1)
    for text in child_texts:
        if text == None:
            continue
        new_state = prompt + ' ' + text
        child = Node(new_state, depth+1)
        child.parent = node
        node.add_child(child)
        if is_leaf_next:
            child.update_win_value(scores.get(text))

def main(question, true_sql, db_id, expansion_count, schema_items, mins_timeout = None, prompt = None, step_prompts = None):
    root_node = Node(prompt, depth=1)
    root_node.visits = 1
    root_node.win_value = 0
    montecarlo = MonteCarlo(root_node, mins_timeout)
    montecarlo.db = db_id
    montecarlo.sql = true_sql
    montecarlo.prompt = prompt
    montecarlo.step_prompts = step_prompts
    montecarlo.child_finder = child_finder

    montecarlo.simulate(true_sql, db_id, schema_items, expansion_count)

    stats(montecarlo)
    final_stat = montecarlo.get_stat_dict()
    state_value_list = montecarlo.get_vlue_and_state()
    print("""SOLUTIONS:\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}""".format(db_id, question, true_sql, str(list(set(montecarlo.solutions))), str(list(set(montecarlo.negatives))), json.dumps(final_stat), json.dumps(state_value_list), str([prompt])))

if __name__ == "__main__":
    for problem in problems:
        (
            prompt,
            max_new_tokens,
            expansion_count,
            max_depth,
            task,
            query,
            db_id,
            question,
            schema_items,
            step_prompts
        ) = problem

        main(question, query, db_id, expansion_count, schema_items, prompt=prompt, step_prompts=step_prompts)
