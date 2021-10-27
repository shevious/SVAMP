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
    # n-1: punctuate(maybe), n-2: last word, n-3: last-1 pos word
    for i in range(n - 3, -1, -1):
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
        q = replace_num_words(q)
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
        return '(%s %% %s)' % (n1, n2)
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
    eq += "print(('%.2f'%"+calc_eq(eq_list)+").rstrip('0').rstrip('.'))\n"
    ans = ('%.2f'%val).rstrip('0').rstrip('.')
    return ans, eq


from sympy import symbols, sympify, linsolve, solve


def eval_solution(concrete_solution):
    concrete_solution_eval = list(map(lambda x: bool(x >= 0) and bool(x < 10) and int(x) == x, concrete_solution))
    if sum(concrete_solution_eval) == len(concrete_solution):
        return True

    return False


def enumerate_solutions(Sols):
    if Sols.free_symbols == set():  # no free variables. see if all variables belong to {0,1}

        concrete_solution = Sols.args[0]
        if concrete_solution == set():  # no solutions
            return []
        if eval_solution(concrete_solution):
            return [concrete_solution]
        else:
            return []
    # create a list of tuples of free variables, one for each valid value
    free_vars = []
    for i in range(10 ** len(Sols.free_symbols)):
        free_vars.append(tuple(Sols.free_symbols))

    # generate values to substitute for free variables
    # free_vals = [list(bin(i))[2:] for i in range(10**len(Sols.free_symbols))]
    # free_vals = [tuple(map(int, list('0'*(len(Sols.free_symbols)-len(s)))+s )) for s in free_vals]
    n = len(Sols.free_symbols)
    free_vals = []
    for i in range(10 ** n):
        num_list = []
        for j in range(n):
            num_list.append((int)((i // 10 ** (n - 1 - j)) % 10))
        free_vals.append(tuple(num_list))

    # zip twice to generate lists of pairs of variable and value
    free_zip = zip(free_vars, free_vals)
    free_zip_fin = list([list(zip(x[0], x[1])) for x in free_zip])

    correct_solutions = []

    for substitution in free_zip_fin:
        concrete_solution = list(map(lambda x: x.subs(substitution), Sols.args[0]))
        if eval_solution(concrete_solution):
            correct_solutions.append(concrete_solution)

    return correct_solutions


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
    # print(new_eqs)
    try:
        sol = solve(new_eqs)
        sol_dict = {}
        # print(sol)
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
    var = symbols('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z x', integer=True)
    new_eqs = []
    for eq in eqs:
        terms = eq.split('=')
        for i in range(len(terms) - 1):
            new_eqs.append('(' + terms[i] + ')-(' + terms[i + 1] + ')')
        # new_eqs.append('x-('+x+')')
    exp_eqs = []
    for eq in new_eqs:
        match_list = re.findall(r'[A-Z0-9]+', eq)
        for match in match_list:
            if is_number(match):
                continue
            n = len(match)
            new_eq = ''
            for i, c in enumerate(match):
                if i < n - 1:
                    mul = '*1' + '0' * (n - 1 - i) + '+'
                else:
                    mul = ''
                new_eq += c + mul
            new_eq = '(' + new_eq + ')'
            eq = eq.replace(match, new_eq, 1)
        exp_eqs.append(eq)

    symbol_list = []
    for eq in exp_eqs:
        match_list = re.findall(r'[A-Z]', eq)
        for match in match_list:
            if match not in symbol_list:
                symbol_list.append(match)
    # print(exp_eqs)
    # if True:
    try:
        var_list = [sympify(s) for s in symbol_list]
        res = linsolve(exp_eqs, var_list)
        # print(res)
        l = enumerate_solutions(res)
        # print(l)
        if len(l) == 0:
            raise
        sol = l[0]
        fin_eqs = []
        for i, s in enumerate(symbol_list):
            fin_eqs.append('%s-%i' % (s, sol[i]))
        fin_eqs.append('x-%s' % x)

        sol = solve(fin_eqs)
        sol_dict = {}
        # print(sol)
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


def solve_formula(q):
    match_list = re.findall(r'[A-Z\+\-=\/\*\(\)0-9]+', q)
    # print(match_list)
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
        pos_list = re.findall(r'(\d[A-Z]|[A-Z]\d|[A-Z][A-Z])', q)
        # print(eq, pos_list)
        if len(pos_list) > 0:
            is_pos_problem = True
            break
    if is_pos_problem:
        return solve_pos(eqs, x_list[-1])
    else:
        return solve_eqs(eqs, x_list[-1])


def rel_str(n, seq_n, seq_n_1, seq_type, rels):
    rel = rels[seq_type]
    rel = rel.replace('n', str(n))
    rel = rel.replace('a_i', seq_n)
    rel = rel.replace('a_j', seq_n_1)
    return rel


def parse_sol(sol, eq_symbol_list, seq_type):
    if type(sol) is list:
        if len(sol) == 0:
            return None
        sol = sol[0]
    sol_dict = {}
    if type(sol) is dict:
        for key in sol.keys():
            sol_dict[str(key)] = str(sol[key])
            if not is_number(str(sol[key])):
                return None
    elif type(sol) is tuple:
        for i, val in enumerate(sol):
            sol_dict[eq_symbol_list[i]] = str(val)
    sol_dict['seq_type'] = seq_type
    return sol_dict


def fun0(a, b, i, c):
    return a + b * i + c


def fun1(a, b, i, c):
    return a * b + c


def fun2(a, b, i, c):
    return a + (i + b) ** 2 + c


def calc_seq(sol_dict, n, n_id, seq_type, rels):
    calc_str = [
        'a+b*i+c',
        'a*b+c',
        'a+(i+b)**2+c'
    ]

    funs = [fun0, fun1, fun2]

    b = int(sol_dict['b'])
    c = int(sol_dict['c'])

    a = int(sol_dict['a_0'])
    for i in range(0, int(n) - 1):
        a = funs[seq_type](a, b, i, c)
    ans = a
    py_eq = 'a = %s\n' % (sol_dict['a_0'])
    py_eq += 'b = %s\n' % (sol_dict['b'])
    py_eq += 'c = %s\n' % (sol_dict['c'])
    py_eq += 'for i in range(0,%s-1):\n' % n
    py_eq += '    a = %s\n' % calc_str[seq_type]
    py_eq += '%s = a\n' % n_id
    return ans, py_eq


def create_ans(sol_dict, num_list, formula, var_list, seq_type, rels):
    failed = (None, 'print(1)')
    if formula is not None:
        py_eq = "formula = '%s'\n" % rels[seq_type]
        for var in sol_dict.keys():
            if var == 'seq_type':
                continue
            if not is_number(sol_dict[var]):
                return failed
            if var == 'x':
                py_eq += '%s = %s\n' % (var, formula)
            else:
                py_eq += '%s = %s\n' % (var, sol_dict[var])
        py_eq += 'print(x)\n'
        ans = sol_dict['x']
        return ans, py_eq
    ans1, py_eq1 = calc_seq(sol_dict, num_list[0], 'num1', seq_type, rels)
    if len(num_list) == 1:
        ans = ans1
        py_eq = py_eq1 + 'print(num1)\n'
        return str(ans), py_eq
    ans2, py_eq2 = calc_seq(sol_dict, num_list[1], 'num2', seq_type, rels)
    ans = ans2 - ans1
    py_eq = py_eq1 + py_eq2
    py_eq += 'print(num2-num1)\n'
    return str(ans), py_eq


import io, contextlib
import runpy
from contextlib import redirect_stdout

def check_ans(ans, py_eq):
    with open("run_sol.py", "w") as text_file:
        text_file.write(py_eq)

    with io.StringIO() as buf, redirect_stdout(buf):
        runpy.run_path(path_name='run_sol.py')
        output = buf.getvalue()
    output = output.strip('\n')
    if output == ans:
        return True
    else:
        return False


def seq_proc(seq, num_list, var_list, formula):
    failed = (None, 'print(0)')
    var = symbols('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z x', integer=True)
    var = symbols('a_0 b c', integer=True)
    rels = [
        'a_i + b*n + c - a_j',  # a[n+1] = b*n + c + a[n]
        'a_i*b + c - a_j',  # a[n+1] = b*a[n] + c
        'a_i + (n+b)**2 + c - a_j',  # a[n+1] = (n+b)**2 + c + a[n]
    ]
    symbol_list = [
        ['a_0', 'b', 'c'],
        ['a_0', 'b', 'c'],
        ['a_0', 'b', 'c'],
    ]
    n_seq_type = 3
    n_seq = len(seq)
    is_success = False
    for seq_type in range(n_seq_type):
        eqs = ['a_0 - (%s)' % seq[0]]
        for n in range(n_seq - 1):
            eqs.append(rel_str(n, seq[n], seq[n + 1], seq_type, rels))
        eq_symbol_list = symbol_list[seq_type] + var_list
        if formula is not None:
            eqs.append('x - (%s)' % formula)
            eq_symbol_list.append('x')
        # print(eqs)
        try:
            sol = solve(eqs, eq_symbol_list)
            sol_dict = parse_sol(sol, eq_symbol_list, seq_type)
            # print(eqs)
            # print('sol_dict')
            # print(sol_dict)
            if sol_dict is None:
                continue
            ans, py_eq = create_ans(sol_dict, num_list, formula, var_list, seq_type, rels)
            if ans is None:
                continue
            # print(ans)
            # print(py_eq)
            is_success = True
            return ans, py_eq
        except:
            continue

    if not is_success:
        return failed


def solve_seq(q):
    failed = (None, 'print(0)')
    match_list = re.findall(r'((\d+|[A-Z]), (\d+|[A-Z]), (\d+|[A-Z]), (\d+|[A-Z])(, (\d+|[A-Z]))+)', q)
    if len(match_list) == 0:
        return failed
    seq_str = match_list[0][0]
    seq = seq_str.split(', ')
    var_list = []
    for a in seq:
        if is_number(a):
            continue
        else:
            match = re.fullmatch(r'[A-Z]', a)
            if match is None:
                return failed
            if a not in var_list:
                var_list.append(a)
    pos = q.find(seq_str) + len(seq_str)
    remained = q[pos:]
    num_list = re.findall(r'\d+', remained)
    formula_list = re.findall(r'[A-Z\-\+]+', remained)
    if len(var_list) > 0:
        if len(formula_list) == 0:
            return failed
        if len(var_list) > 2:
            return failed
        formula = formula_list[-1]
    elif len(num_list) > 0:
        num_list = num_list[-2:]
        formula = None
    else:
        return failed
    # try:
    if True:
        return seq_proc(seq, num_list, var_list, formula)
    # except:
    # return failed


agc_labels_0 = ['남준', '석진', '윤기', '호석', '지민', '태형', '정국', '민영', '유정', '은지', '유나']
agc_labels_1 = ['(가)', '(나)', '(다)', '(라)', '(마)', '(바)']
agc_labels_2 = ['흰색', '검정색', '빨간색', '파란색', '노란색', '초록색']
agc_labels_3 = ['사과', '복숭아', '배', '참외', '감', '귤', '포도', '수박']
agc_labels_4 = ['오리', '닭', '토끼', '물고기', '고래', '거위', '달팽이', '개구리', '강아지', '고양이', '비둘기', '병아리']
agc_labels = [agc_labels_0, agc_labels_1, agc_labels_2, agc_labels_3, agc_labels_4]

person_q_tokens = ['누구', '누가']
label_q_tokens = ['무엇', '어느']
number_q_tokens = ['몇', '구하시오', '얼마']

number_words = [r'(^|\s)(한)(\s)', r'(^|\s)(두)(\s)', r'(^|\s)(세)(\s)', r'(^|\s)(네)(\s)']
number_words_c = [r'\g<1>1\3', r'\g<1>2\3', r'\g<1>3\3', r'\g<1>4\3']
number_words += [r'(^|\s)(하나)', r'(^|\s)둘', r'(^|\s)셋', r'(^|\s)넷']
number_words_c += [r'\g<1>1', r'\g<1>2', r'\g<1>3', r'\g<1>4']
number_words += [r'(^|\s)(다섯)', r'(^|\s)여섯', r'(^|\s)일곱', r'(^|\s)여덟']
number_words_c += [r'\g<1>5', r'\g<1>6', r'\g<1>7', r'\g<1>8']
number_words += [r'(^|\s)(아홉)', r'(^|\s)열']
number_words_c += [r'\g<1>9', r'\g<1>10']

def replace_num_words(q):
    for i, words in enumerate(number_words):
        q = re.sub(words, number_words_c[i], q)
    return q

def is_label_p(q):
    is_number = False
    for token in number_q_tokens:
        match = re.findall(token, q)
        if len(match) > 0:
            is_number = True
            break
    if is_number:
        return None

    is_person = False
    is_label = False
    for token in person_q_tokens:
        match = re.findall(token, q)
        if len(match) > 0:
            is_person = True
            break
    for token in label_q_tokens:
        match = re.findall(token, q)
        if len(match) > 0:
            is_label = True
            break
    if not is_person and not is_label:
        return None

    cnts = [0] * len(agc_labels)
    for i, labels in enumerate(agc_labels):
        if i == 0 and not is_person:
            continue
        if i != 0 and not is_label:
            continue
        for token in labels:
            token_new = token.replace('(', '\\(').replace(')', '\\)')
            match = re.findall(token_new, q)
            cnts[i] += len(match)
    max_cnt = max(cnts)
    if max_cnt <= 1:
        return None
    max_i = cnts.index(max_cnt)

    label_tuples = []
    for token in agc_labels[max_i]:
        pos = q.find(token)
        if (pos == -1):
            continue
        label_tuples.append((pos, token))
    labels = [label for _, label in sorted(label_tuples)]
    return labels

def answer_label(ans, labels):
    n = len(labels)
    ans = int(float(ans))
    if ans < 0 or ans >= n:
        ans = 0
    py_eq = 'labels = ['
    for label in labels:
        py_eq += "'"+label+"',"
    py_eq += ']\n'
    py_eq += 'print(labels[%i])\n'%ans
    return labels[ans], py_eq
