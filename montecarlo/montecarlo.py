import random
import time
import ipdb
import copy
import re
import time
import llm
from cmdline import args
from sql import score_func, normalize_sql
from common import limit_depth
from utils import check_consistency

class MonteCarlo:
    def __init__(self, root_node, mins_timeout=None):
        self.root_node = root_node
        self.solutions = set()
        self.negatives = set()
        self.child_finder = None
        self.node_evaluator = lambda child, montecarlo: None
        self.stats_expansion_count = 0
        self.stats_failed_expansion_count = 0
        self.mins_timeout = mins_timeout
        self.db = None
        self.sql = ''
        self.prompt = ''
        self.step_prompts = []

    def simulate(self, true_sql, db_id, schema_items, expansion_count=1):
        i = 0

        start_time = time.time()

        while expansion_count is None or i < expansion_count:
            i += 1
            if self.solutions is not None and len(self.solutions) > 10:
                return

            if self.mins_timeout is not None:
                curr_time = time.time()
                duration = curr_time - start_time

                if duration > (self.mins_timeout * 60):
                    print("reached timelimit, stopping expansion on current node")
                    return

            current_node = self.root_node
            
            while current_node.expanded:
                if len(current_node.children) > 0:
                    current_node = current_node.get_preferred_child(self.root_node)
                else:
                    break
            self.expand(current_node, true_sql, db_id, schema_items)

    def expand(self, node, true_sql, db_id, schema_items):
        if node.expanded:
            return
        self.stats_expansion_count += 1
        self.child_finder(node, true_sql, db_id, schema_items, self)

        if len(node.children) > 0:
            node.expanded = True
            rand_child = random.choice(node.children)
            child_win_value = self.rollout(rand_child, schema_items)
            rand_child.update_win_value(child_win_value)
        else:
            node.expanded = True
            self.stats_failed_expansion_count += 1

    def random_rollout(self, node):
        self.child_finder(node, self.sql, self.db, self)
        if len(node.children) == 0:
            child_win_value = score_func(node.state, self.sql, self.db)
            return child_win_value
        child = random.choice(node.children)

        if limit_depth(child):
            child_win_value = -2
            return child_win_value
        child_win_value = score_func(child.state, self.sql, self.db)
        if child_win_value == 1:
            self.solutions.add(child.state)
        
        if child_win_value == 0:
            self.negatives.add(child.state)
        
        if child_win_value != -1:
            return child_win_value
        else:
            return self.random_rollout(child)
        
    def rollout(self, node, schema_items):
        sql_texts = list()
        scores = list()
        state = node.state
        depth = node.depth
        prompt = '{}\n{}'.format(state, self.step_prompts[-1])
        beam_search_num = args.beam_search_num
        if args.infer_methd == 'vllm':
            texts = llm.generate_by_vllm(state, beam_search_num)
        else:
            texts = llm.generate(state, beam_search_num)

        for text in texts:
            sql_text = text
            sql_pattern = r"```sql\n(.*?)\n```"
            sql_match = re.search(sql_pattern, sql_text, re.DOTALL)
            if sql_match:
                sql_text = next((group for group in sql_match.groups() if group is not None), None)
                sql_text = sql_text.strip().replace('\n', ' ').strip()
                sql_text = sql_text.replace("```", '').strip()
                sql_text = sql_text.strip().replace('\n', ' ').strip()
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
                    else:
                        sql_pattern = r"```\n(.*?)\n```"
                        sql_match = re.search(sql_pattern, sql_text, re.DOTALL)
                        if sql_match:
                            sql_text = next((group for group in sql_match.groups() if group is not None), None)
                            sql_text = sql_text.replace("```", '').strip()
                        else:
                            match = re.search(r'query:(.*)', text)
                            if match:
                                result = match.group(1).strip()
                            else:
                                sql_text = sql_text.split(';')[0]
                    sql_text = sql_text.strip().replace('\n', ' ').strip()  
                except:
                    pass

            is_consistency = check_consistency(prompt, sql_text, match_level=depth)
            if not is_consistency:
                continue

            if len(sql_text) > 300:
                score = -2
                sql_text = None
            else:
                score = score_func(sql_text, self.sql, self.db)
            scores.append(score)
            sql_texts.append(sql_text)
        if 1 in scores:
            win_value = 1
        else:
            win_value = 0
        node.update_win_value(win_value)
        return win_value

    def print_tree(self, f):
        f.write("graph\n{\n")
        self.root_node.print_node(f, 0, self.root_node, "a")
        f.write("}\n")

    def get_widths(self):
        widths = [1]
        nodes = [self.root_node]
        while any([len(n.children) > 0 for n in nodes]):
            new_nodes = []
            for node in nodes:
                for child in node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            widths.append(len(nodes))
        return widths

    def get_child_counts(self):
        counts = [1]
        nodes = [self.root_node]
        while any([len(n.children) > 0 for n in nodes]):
            new_nodes = []
            for node in nodes:
                for child in node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            counts.extend([len(n.children) for n in nodes])
        return counts

    def get_values_and_visits(self):
        values = [self.root_node.win_value]
        visits = [self.root_node.visits]
        expected_values = [self.root_node.win_value / self.root_node.visits]
        nodes = [self.root_node]
        while any([len(n.children) > 0 for n in nodes]):
            new_nodes = []
            for node in nodes:
                for child in node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            values.extend([n.win_value for n in nodes])
            visits.extend([n.visits for n in nodes])
            expected_values.extend([n.win_value / (n.visits or 1) for n in nodes])
        return values, visits, expected_values

    def get_widen_count(self):
        count = 0
        nodes = [self.root_node]
        while any([len(n.children) > 0 for n in nodes]):
            new_nodes = []
            for node in nodes:
                for child in node.children:
                    new_nodes.append(child)
            nodes = new_nodes
            count += len([n for n in nodes if n.is_widen_node])
        return count

    def get_stat_dict(self):
        stat = {}
        widths = self.get_widths()
        stat["width"] = max(widths)
        stat["depth"] = len(widths)
        stat["total_nodes"] = sum(widths)

        child_counts = self.get_child_counts()
        stat["mean_child_count"] = sum(child_counts) / len(child_counts)
        stat["max_child_count"] = max(child_counts)
        stat["leaf_node_count"] = len([1 for c in child_counts if c == 0])

        values, visits, expected_values = self.get_values_and_visits()
        stat["mean_value"] = sum(values) / len(values)
        stat["max_value"] = max(values)
        stat["min_value"] = min(values)
        stat["mean_visits"] = sum(visits) / len(visits)
        stat["max_visits"] = max(visits)
        stat["min_visits"] = min(visits)
        stat["mean_expected_value"] = sum(expected_values) / len(expected_values)
        stat["max_expected_value"] = max(expected_values)
        stat["min_expected_value"] = min(expected_values)

        return stat


    def get_vlue_and_state(sefl, f=lambda x: x):
        queue = [sefl.root_node]
        state_value_list = []
        n_nodes = 0
        while queue != []:
            node = queue.pop()
            n_nodes += 1
            is_back = node.parent is not None and f(node.state) == f(node.parent.state)
            is_leaf = node.children == []
            if is_back:
                pass
            if is_leaf:
                if is_back:
                    pass
            if node.state != '':
                node_data = {'state': node.state, 'win_value': node.win_value, 'visits': node.visits, 'is_leaf': is_leaf, 'depth': node.depth}
                state_value_list.append(node_data)
            queue += node.children

        return state_value_list