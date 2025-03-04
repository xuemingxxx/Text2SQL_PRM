import os
import re
import json
import nltk
import numpy as np
import tqdm
import sqlite3
import ipdb
from cmdline import args
from collections import defaultdict
from db_utils import get_db_schema_sequence, get_matched_content_sequence


def read_schema_detail(database_root, db_id):
    schema = {}
    db_path = f'{database_root}/{db_id}/{db_id}.sqlite'
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}'".format(table_name))
            create_sql = cursor.fetchall()[0][0]
            # print(database_root, db_id, table_name)
            cursor.execute("SELECT * FROM `{}` LIMIT 2".format(table_name))
            demos = cursor.fetchall()
            cursor.execute(f"PRAGMA table_info(`{table_name}`);")
            results = cursor.fetchall()
            schema[table_name] = dict()
            schema[table_name]['column_info'] = [[r[1],r[2], create_sql] for r in results]
            schema[table_name]['creat_sql'] = re.sub(r'\s+', ' ', create_sql)
            schema[table_name]['demos'] = [(str(d) for d in demo) for demo in demos]
    return schema

def generate_prompt_detail(database_root, db_id, question, target_seq, data, do_example=False, retriever_gen=None, step_prompts=None):
    schema = read_schema_detail(database_root, db_id)
    question = data['question']
    PRE_PROMPT = 'Please finish the task: Convert Question into an appropriate SQL query based on the detailed database schema and demo data.\n\n'
    TEMPLATE = "-- Database {}\n{}\n-- Question: {}\n"
    if True:
        schema_with_comment = data['schema']['schema_items']
        for t in schema_with_comment:
            t['column_names'] = ['`{}`'.format(name) if len(name.split(' '))!=1  else name for name in t['column_names']]
        schema_text = '\n\n'.join(['\n'.join([table_schema['creat_sql'], 'DEMOS:\n'+'\n'.join(['|'.join(data) for data in table_schema['demos']])]) for table_name, table_schema in schema.items()])

        prefix_seq = TEMPLATE.format(data['db_id'], schema_text, question)
    AFTER_PROMPT = """\nPlease generate the SQL query for the given question. 
        -- Please thinking step by step for generating the SQL. 
        ## Step 1: identifying the table and columns that are useful for genenrating the queries.
        ## Step 2: if it requires the sub-queries, please generate the sub-queries first.
        ## Step 3: generating the conditions for the queries. 
        ## Step 4: list only the key important columns information for answering the question.
        ## Step 5: Generating the SQL with the key important columns to display, and use the above sub-queries and conditions."""

    prefix_seq = PRE_PROMPT + prefix_seq
    return prefix_seq

count = 0
text_to_sql_problems = list()
if args.dataset_name == 'spider':
    # database_root = './datasets/spider/test_database'
    # data_file = './datasets/spider_dev_text2sql.json'
    database_root = './datasets/spider/database'
    data_file = './datasets/spider_train_text2sql.json'

    with open(data_file) as f:
        dataset = json.loads(f.read().strip())

    print('begin_count: 0')
    for data in dataset:
        db_id = data['db_id']
        data["schema_sequence"] = get_db_schema_sequence(data["schema"])
        data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        question = data['question']
        query = data['sql']
        schema_items = data["schema"]["schema_items"]

        step_prompts = ["", "Step 1: Please analyze the tables that are relevant for generating the  SQL. After completing the analysis, list only the used tables with the format: {'tables': ['table1', 'table2']}.\n", "Step 2: Please analyze columns that will be useful for generating the SQL.  After completing the analysis, list only the useful columns in the following format: {'columns': ['column1', 'column2']}, without further analysis.\n", "Step 3:  Analyze and define the join conditions required for generating the SQL query. After completing the analysis, list only the conditions in the following format: {'conditions': ['condition1', 'condition2']}, without further analysis.\n", "Step 4: Based on the provided schema, directly generate the SQL query for the given question by leveraging identified relevant tables, columns, and conditions, without repeating step-by-step analysis.\n"]
        
        prompt = generate_prompt_detail(database_root, db_id, question, query, data, step_prompts=step_prompts)
        max_depth = len(step_prompts)
        count = count + 1
        text_to_sql = (
            '''{}'''.format(prompt),
            args.max_new_tokens,
            args.expansion_count,
            max_depth,
            'SQL',
            query,
            db_id,
            question,
            schema_items,
            step_prompts)
        text_to_sql_problems.append(text_to_sql)
    
elif args.dataset_name == 'bird':
    # database_root = './datasets/bird/dev/dev_databases/'
    # data_file = '../datasets/bird_dev_text2sql.json'
    database_root = './datasets/bird/train/train_databases/'
    data_file = './datasets/bird_train_text2sql.json'
    with open(data_file) as f:
        dataset = json.loads(f.read().strip())
    print('datset size:', len(dataset))
    print('begin_count: 0')
    for data in dataset:
        db_id = data['db_id']
        question = data['question']
        query = data['sql']
        schema_items = data["schema"]["schema_items"]

        step_prompts = ["", "Step 1: Please analyze the tables that are relevant for generating the  SQL. After completing the analysis, list only the used tables with the format: {'tables': ['table1', 'table2']}.\n", "Step 2: Please analyze columns that will be useful for generating the SQL.  After completing the analysis, list only the useful columns in the following format: {'columns': ['column1', 'column2']}, without further analysis.\n", "Step 3:  Analyze and define the join conditions required for generating the SQL query. After completing the analysis, list only the conditions in the following format: {'conditions': ['condition1', 'condition2']}, without further analysis.\n ", "Step 4: Based on the provided schema, directly generate the SQL query for the given question by leveraging identified relevant tables, columns, and conditions, without repeating step-by-step analysis.\n"]

        prompt = generate_prompt_detail(database_root, db_id, question, query, data, step_prompts=step_prompts)
        count = count + 1
        max_depth = len(step_prompts)
        text_to_sql = (
            '''{}'''.format(prompt),
            args.max_new_tokens,
            args.expansion_count,
            max_depth,
            'SQL',
            query,
            db_id,
            question,
            schema_items,
            step_prompts)
        text_to_sql_problems.append(text_to_sql)

print('Total Task Count', len(text_to_sql_problems))
print('\n\nTask 0:')
print(text_to_sql_problems[0][0])
print('\n\nTask 1:')
print(text_to_sql_problems[1][0])

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
) = text_to_sql_problems[0]
problems =  text_to_sql_problems