import json
import pandas as pd
import re


def clean_string(floatString):
    return floatString.rstrip('0').rstrip('.')


def convert_frac(q):
    match_list = re.findall(r'\d+/\d+', q)
    for match_str in match_list:
        new_str = clean_string(str(eval(match_str)))
        q = q.replace(match_str, new_str)
    return q


def pre_question(q):
    q = convert_frac(q)
    return q


def break_punctuate(q):
    q = re.sub(r'([^\d])(\.)($|\s)', r'\1 \2\3', q)
    q = re.sub(r'([^\s])([\,\?\'\"\:])', r'\1 \2', q)
    q = re.sub(r'([\,\?\'\"\:])([^\s])', r'\1 \2', q)
    return q


def convert_number(q):
    match_list = re.findall(r"[-+]?\d*\.\d+|\d+", q)
    num_list = []
    # skip for safety
    if 'number' in q:
        return ' '.join(num_list), q

    for i, match in enumerate(match_list):
        q = q.replace(match, 'number_', 1)
        s = match
        if '.' not in s:
            s = s + '.0'
        num_list.append(s)
    for i, match in enumerate(match_list):
        q = q.replace('number_', 'number%i' % i, 1)
    return ' '.join(num_list), q


def isnot_punc(word):
    return word not in ['.', ',', '?']


def make_group_num(q):
    words = q.split()
    n = len(words)
    group_num = []
    for i, word in enumerate(words):
        if 'number' in word:
            for j in range(i - 1, -1, -1):
                if isnot_punc(word):
                    if j not in group_num:
                        group_num.append(j)
                    break
            if i not in group_num:
                group_num.append(i)
            for j in range(i + 1, n):
                if isnot_punc(word):
                    if j not in group_num:
                        group_num.append(j)
                    break
    last_num = 0

    '''
    for i in range(n-3, -1, -1):
        if isnot_punc(words[i]):
            last_num += 1
            if i not in group_num:
                group_num.append(i)
            if last_num >= 3:
                break
    '''
    for i in range(n - 2, -1, -1):
        last_num += 1
        group_num.append(i)
        if last_num >= 3:
            break
    group_num.sort()
    return '%s' % group_num


def load_infer_data(path):
    with open(path, "r") as json_file:
        q_json = json.load(json_file)
    n_question = len(q_json.keys())
    df = pd.DataFrame(columns=['Question', 'Numbers', 'group_nums', 'id'])
    for i, key in enumerate(q_json.keys()):
        q = q_json[key]['question']
        q_new = break_punctuate(q)
        q_new = pre_question(q_new)
        nums, q_new = convert_number(q_new)
        group_num = make_group_num(q_new)
        df = df.append({
            'Question': q_new,
            'Question_org': q,
            'Numbers': nums,
            'group_nums': eval(group_num),
            'id': key
        }, ignore_index=True)
    return df.to_dict('records')

def calc_eq(eq_list):
    node = eq_list.pop(0)
    match = re.search(r"[-+]?\d*\.\d+|\d+", node)
    if match is not None:
        return match.group(0)
    n1 = calc_eq(eq_list)
    n2 = calc_eq(eq_list)
    if node == '*':
        return '(%s * %s)' % (n1, n2)
    elif node == '/':
        return '(%s / %s)' % (n1, n2)
    elif node == '+':
        return '(%s + %s)' % (n1, n2)
    elif node == '-':
        return '(%s - %s)' % (n1, n2)
    elif node == '//':
        return '(%s // %s)' % (n1, n2)
    elif node == '%':
        return '(%s % %s)' % (n1, n2)
    elif node == 'P':
        return '(math.factorial(%s)/math.factorial(%s-%s))' % (n1, n1, n2)
    elif node == 'C':
        return '(math.factorial(%s)/(math.factorial(%s-%s)*math.factorial(%s))' % (n1, n1, n2, n2)
    elif node == 'H':
        return '(math.factorial(%s+%s-1)/(math.factorial(%s+%s-1-%s)*math.factorial(%s)))' % (n1, n2, n1, n2, n2, n2)
    else:
        return None

def convert_eq(val, eq_list):
    if val is None:
        ans = "0"
        py_eq = "print(0)"
        return ans, py_eq
    #print(eq_list)
    eq = 'import math\n'
    eq += 'print(("%.2f"%'+calc_eq(eq_list)+').rstrip("0").rstrip("."))\n'
    ans = ('%.2f'%val).rstrip("0").rstrip(".")
    return ans, eq


from sympy.solvers import solve
from sympy import symbols


def is_number(q):
    if re.match(r'^[-+]?(\d*\.\d+|\d+)$', q) is None:
        return False
    else:
        return True


def solve_eqs(eqs, x):
    var = symbols('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z x')
    new_eqs = []
    for eq in eqs:
        terms = eq.split('=')
        for i in range(len(terms) - 1):
            new_eqs.append('(' + terms[i] + ')-(' + terms[i + 1] + ')')
    new_eqs.append('x-(' + x + ')')
    print(new_eqs)
    try:
        sol = solve(new_eqs)
        sol_dict = {}
        print(sol)
        for var in sol.keys():
            sol_dict[str(var)] = str(sol[var])
        for var in sol_dict.keys():
            if not is_number(sol_dict[var]):
                raise
        if 'x' not in sol_dict.keys():
            raise

        ans = sol_dict['x']
        equation = ''

        for var in sol_dict.keys():
            equation += var + '=' + sol_dict[var] + '\n'
        equation += 'print(' + 'x' + ')\n'
        return ans, equation
    except:
        return None, 'print(0)'


def solve_pos(eqs, x):
    var = symbols('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z x')
    return None, 'print(0)'


def solve_formula(q):
    match_list = re.findall(r'[A-Z\+\-=\/\*\(\)0-9]+', q)
    eqs = []
    x_list = []
    for match in match_list:
        if '=' in match:
            eqs.append(match)
        else:
            x_list.append(match)
    if len(eqs) == 0 or len(x_list) == 0:
        return None, 'print(0)'
    is_pos_problem = False
    for eq in eqs:
        pos_list = re.findall(r'(\d[A-Z]|[A-Z]\d)', q)
        if len(pos_list) > 0:
            is_pos_problem = True
            break
    if is_pos_problem:
        return solve_pos(eqs, x_list[-1])
    else:
        return solve_eqs(eqs, x_list[-1])