import re
import json
import ipdb
from sql_metadata import Parser

def check_consistency(pre_state, pre_sql, match_level=3):
    is_consistency = False
    try:
        parser = Parser(pre_sql)
        columns_in_sql = parser.columns
        columns_in_sql = list(set([column.split('.')[-1] for column in columns_in_sql])) 
        tables_in_sql = parser.tables
        table_pattern = r"'tables': \[([^\]]+)\]"
        column_pattern = r"'columns': \[([^\]]+)\]"
        if match_level >= 2:
            tables_in_state = re.findall(table_pattern, pre_state)[1].split(',')
            tables_in_state = [table.replace("'", "").strip() for table in tables_in_state]
            tables_in_state = list(set([table.strip().strip("'").strip() for table in tables_in_state])) 
            table_is_equal = set(item.lower() for item in tables_in_sql) == set(item.lower() for item in tables_in_state)
        if match_level >= 3:
            columns_in_state = re.findall(column_pattern, pre_state)[1].split(',')
            columns_in_state = [column.replace("'", "").strip() for column in columns_in_state]
            columns_in_state = list(set([column.strip().strip("'").strip().split('.')[-1] for column in columns_in_state])) 
            column_is_equal = set(item.lower() for item in columns_in_sql) == set(item.lower() for item in columns_in_state)

        if match_level == 2:
            is_consistency = table_is_equal
        elif match_level >= 3:
            is_consistency = table_is_equal and column_is_equal
    except:
        pass
    return is_consistency

def make_duplication(curr_texts, match_level=2):
    uniq_texts = list()
    if match_level!=2 and match_level!=3:
        return curr_texts

    if True:
        for curr_text in curr_texts:
            is_duplication = False
            for pre_text in uniq_texts:
                # print(pre_text, curr_text)
                try:
                    if match_level == 2:
                        pre_sate = eval(pre_text)['tables']
                        curr_state = eval(curr_text)['tables']
                    if match_level == 3:
                        pre_sate = eval(pre_text)['columns']
                        curr_state = eval(curr_text)['columns']
                    is_duplication = set(item.lower() for item in pre_sate) == set(item.lower() for item in curr_state)
                except:
                    continue
                if is_duplication:
                    break
            if not is_duplication:
                uniq_texts.append(curr_text)
    return uniq_texts