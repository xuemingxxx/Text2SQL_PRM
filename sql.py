from execute import execute
import requests
import re
import os
from typing import Optional
import ipdb
import sqlite3
import sqlparse
from func_timeout import func_set_timeout, FunctionTimedOut
from db_utils import check_sql_executability, detect_special_char
from process_sql import get_schema, Schema, get_sql
from evaluation import Evaluator, build_valid_col_units, rebuild_sql_val, rebuild_sql_col, build_foreign_key_map_from_json
from exec_eval import eval_exec_match
from default_prompts import database_root

evaluator = Evaluator()
kmaps = None
etype = 'exec'
if etype in ['all', 'match', 'exec']:
    # assert args.table is not None, 'table argument must be non-None if exact set match is evaluated'
    kmaps = build_foreign_key_map_from_json('tables.json')

CLAUSE_KEYWORDS = ['select', 'from', 'where', 'group by', 'having', 'order by', 'limit', 'intersect', 'union', 'except', 'union all']
JOIN_KEYWORDS = ['join', 'on', 'as', 'right join', 'inner join', 'left join']
OTHER_KEYWORDS = ['distinct']
BIRD_KEYWORDS = ['if', 'else', 'datediff', 'over', 'instr', 'case', 'partition by', 'iif', 'float', 'real', 'when', 'int', 'using', 'timestampdiff', 'then', 'substr', 'cast', 'integer', 'strftime', 'end']
WHERE_OPS = ['not', 'between', 'in', 'like', 'is', 'exists', 'not null', 'null']
AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']
COND_OPS = ['and', 'or']
ORDER_OPS = ['desc', 'asc']
SQL_KEYWORDS = []
SQL_KEYWORDS.extend(CLAUSE_KEYWORDS)
SQL_KEYWORDS.extend(JOIN_KEYWORDS)
SQL_KEYWORDS.extend(OTHER_KEYWORDS)
SQL_KEYWORDS.extend(BIRD_KEYWORDS)
SQL_KEYWORDS.extend(WHERE_OPS)
SQL_KEYWORDS.extend(AGG_OPS)
SQL_KEYWORDS.extend(COND_OPS)
SQL_KEYWORDS.extend(ORDER_OPS)

def verifier_feedback(ok: str, not_ok: str) -> Optional[str]:
    msg = "Consider previous issue"
    if msg in ok:
        return None
    _, err = calculateScoreHelper(not_ok)
    if err:
        err = err.strip()
        hint = f"\n # {msg}: {err}\n"
        text = ok + hint
        return text
    return None

def calculateScore(msg: str, true_sql: str, db_id: str) -> Optional[float]:
    score, _ = calculateScoreHelper(msg, true_sql, db_id)
    return score

def find_first_index(string, char1, char2):
    try:
        index1 = string.index(char1)
    except ValueError:
        index1 = float('inf')  # If char1 is not found, set to infinity

    try:
        index2 = string.index(char2)
    except ValueError:
        index2 = float('inf')  # If char2 is not found, set to infinity

    if index1 == float('inf') and index2 == float('inf'):
        return -1  # If neither character is found
    else:
        return min(index1, index2)  # Return the minimum index


def exact_result(p_str, g_str, db):
    db_name = db
    db_dir = os.path.join(database_root, db, db + ".sqlite")
    # print(db_dir)
    schema = Schema(get_schema(db_dir))
    try:
        g_sql = get_sql(schema, g_str)
    except Exception as e:
        return -1 
    # hardness = evaluator.eval_hardness(p_sql)
    try:
        p_sql = get_sql(schema, p_str)
    except:
        # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
        p_sql = {
                "except": None,
                "from": {
                    "conds": [],
                    "table_units": []
                },
                "groupBy": [],
                "having": [],
                "intersect": None,
                "limit": None,
                "orderBy": [],
                "select": [
                    False,
                    []
                ],
                "union": None,
                "where": []
                }

    if etype in ["all", "exec"]:
        plug_value = False
        keep_distinct = False
        progress_bar_for_each_datapoint = False
        try:
            exec_score = eval_exec_match(db=db_dir, p_str=p_str, g_str=g_str, plug_value=plug_value,
                                             keep_distinct=keep_distinct, progress_bar_for_each_datapoint=progress_bar_for_each_datapoint)
        except Exception as e:
            exec_score = -1
        if exec_score:
            score = 1
        else:
            score = 0

    if etype in ["all", "match"]:
        # rebuild sql for value evaluation
        kmap = kmaps[db_name]
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
        exact_score = evaluator.eval_exact_match(p_sql, g_sql)
        partial_scores = evaluator.partial_scores
        if exact_score == 0:
            score = 0
        else:
            score = 1
    return score


def calculateScoreHelper(msg: str, true_sql: str, db_id: str) -> (Optional[float], Optional[str]):
    v = filter_code(msg)
    # If the last line is not return, we should keep generating code
    if v == "": 
        return None, None
    v = v.strip()
    db_dir = os.path.join(database_root, db_id, db_id + ".sqlite")
    r = check_sql_executability(v, db_dir)
    if r is not None:
        return -1.0, r
    score = exact_result(v, true_sql, db_id)
    return score, None

def runUnittests(msg: str) -> (Optional[float], Optional[str]):
    v = filter_code(msg + "```").strip()
    if v == "":
        return None, None
    r = check_code(v)
    if r["status"] == 0:
        return 1.0, None
    log = r["log"]
    print(log)
    marker = "ex.py\", line "
    first = log[log.rindex(marker) + len(marker):]
    num_line_first = int(first[0 : find_first_index(first, '\n', ',')])
    if filter_code(msg).strip() != v and num_line_first >= v.count("\n"):
        return None, None
    err = log
    return -1.0, err

def filter_code(msg: str) -> str:
    # m = re.findall("```([Pp]ython)?(.*?)```", msg, re.MULTILINE | re.DOTALL)
    # r = "\n".join([x[1] for x in m])
    m = msg.strip().split('\n')[-1]
    return m

@func_set_timeout(60)
def execute_sql(cursor, sql):
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    return sql_res

def compare_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path, check_same_thread = False)
    # Connect to the database
    cursor = conn.cursor()

    ground_truth_res = execute_sql(cursor, ground_truth)
    try:
        predicted_res = execute_sql(cursor, predicted_sql)
    except Exception as e:
        print("raises an error: {}.".format(str(e)))
        return 0, None, ground_truth_res
    except FunctionTimedOut as fto:
        print("raises an error: time out.")
        return 0, None, ground_truth_res

    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res, predicted_res, ground_truth_res

def check_code(v: str) -> dict:
    execution_error = calculateScoreHelper(v)
    if execution_error is None: # the generated sql has no execution errors, we will return it as the final generated sql
        final_generated_sql = generated_sql

    if final_generated_sql is not None:
            final_generated_sql = "SQL placeholder"

    return execute("python3", "py", v, use_sandbox=True)

test_fwk = """
import sys
runningAllTests = True
def test(x):
	if not x:
		print('FALSE')
		sys.exit(1)
def finalReport():
	global runningAllTests
	if not runningAllTests:
		print('INCONCLUSIVE')
		sys.exit(0)
	else:
		print('TRUE')
		sys.exit(0)
"""
def run_unittests(msg: str, unittest=None):
    v = filter_code(msg + "```").strip()
    file = v
    file += test_fwk
    foundAll = True
    for key, value in unittest.items():
        if v.find(key) != -1:
            file += value + "\n"
        else:
            file += "runningAllTests = False\n"
    file += "\nfinalReport()\n"
    return check_code(file)["status"]

def score_func(sentence: str, true_sql:str,  db_id: str, unittest: Optional[str] = None) -> Optional[float]:
    if len(sentence) == 0:
        score = -2
    else:
        score = calculateScore(sentence, true_sql, db_id)
    # print("Predict SQL:", sentence, "\tSCORE:", score)
    # print('\n')
    return score

def load_sql_keywords(file: str):
    keywords = []
    for key in open(file, 'r').readlines():
        keywords.append(key.strip().upper())
        keywords.append(key.strip().lower())
    return keywords

score_func_whole = score_func
filter_code_whole = filter_code

def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`"+column_name+"`")

    while "``" in sql:
        sql = sql.replace("``", "`")

    return sql


def lexical(query, values):
    if isinstance(query, str):
        for placeholder, value in values.items():
            query = query.replace(placeholder, value)
    elif isinstance(query, list):
        for i in range(len(query)):
            if query[i] in values:
                query[i] = values[query[i]]
    return query


def delexical(query):
    values = {}
    new_query = ""
    in_value = False
    in_col = False
    value = ""
    placeholder_id = 0
    new_query = ""
    for char in query:
        if char == "'":
            in_value = not in_value
            value += char
            if not in_value:
                values[f"value_{placeholder_id}"] = value
                new_query += f"value_{placeholder_id}"
                placeholder_id += 1
                value = ""
        else:
            if not in_value:
                new_query += char
            else:
                value += char
    return new_query, values


def _is_whitespace(sqlparse_token):
    return sqlparse_token.ttype == sqlparse.tokens.Whitespace


def normalize_sql(sql_exp):
    sql_exp = sql_exp.replace('"', "'")
    if sql_exp.count("'") % 2 != 0:  # odd number of single quotes, meaning the value is incomplete or value contains a single quote
        odd_quotes = True
    else:
        odd_quotes = False
    
    if not odd_quotes:
        sql_exp, values = delexical(sql_exp)
        sql_exp = sql_exp.lower()
    
    sql_exp = sql_exp.rstrip(";")
    parse = sqlparse.parse(sql_exp)
    sql = parse[0]
    flat_tokens = sql.flatten()
    sql_tokens = [
        (token.value.upper() if token.value in SQL_KEYWORDS else token.value)
        for token in flat_tokens if not _is_whitespace(token)
    ]

    sql_lower = ' '.join(sql_tokens)
    sql_lower = sql_lower.replace(' . ', '.')
    for op in AGG_OPS:
        sql_lower = sql_lower.replace(f" {op.upper()} (", f" {op.upper()}(")
    sql_lower = sql_lower.replace('( ', '(')
    sql_lower = sql_lower.replace(' )', ')')
    sql_lower = sql_lower.replace(' ,', ',')

    ### BIRD-SQL special cases ###
    sql_lower = sql_lower.replace(' AS text', ' AS TEXT')
    sql_lower = sql_lower.replace(' length(', ' LENGTH(')
    sql_lower = sql_lower.replace(' total(', ' TOTAL(')
    sql_lower = sql_lower.replace(' round(', ' ROUND(')
    ### END ###

    sql_lower = sql_lower.rstrip(";")
    sql_lower += ';'

    if not odd_quotes:
        sql_tokens = lexical(sql_tokens, values)
        sql_lower = lexical(sql_lower, values)
    # else:
    #     print("Cannot process the following SQL")
    #     print(sql_exp, sql_tokens)

    return sql_lower
