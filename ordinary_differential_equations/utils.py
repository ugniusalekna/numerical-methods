import time
import numpy as np
from functools import wraps


def timing(f):
    @wraps(f)
    def wrapper(*a, **kw):
        ts = time.time()
        out = f(*a, **kw)
        te = time.time()
        t = te - ts
        print(f"Runtime of '{f.__name__}': {t:.6f} seconds.")
        return out
    return wrapper


def print_iteration_table(data, m=5, show_vectors=True):
    tot = len(data)
    
    keys = list(data[0].keys())
    column_widths = {}
    
    def format_value(value, width):
        if isinstance(value, (list, np.ndarray)) and show_vectors:
            return f"[{', '.join([f'{v:.5f}' for v in value])}]".ljust(width)
        elif isinstance(value, (int, float)):
            return f"{value:<{width}.8e}"
        return str(value).ljust(width)

    def print_row(i, row_data):
        row = f"{i:<10}"
        for key in keys:
            if key not in column_widths:
                continue
            row += f" {format_value(row_data[key], column_widths[key])}"
        print(row)
    
    header = f"{'step':<10}"
    for key in keys:
        if isinstance(data[0][key], (list, np.ndarray)) and show_vectors:
            width = len(format_value(data[0][key], width=0)) + 10
        elif isinstance(data[0][key], (list, np.ndarray)) and not show_vectors:
            continue
        else:
            width = 15
        column_widths[key] = width
        header += f" {key:<{width}}"
    
    print(header)
    print('-' * len(header))
    
    if tot <= 2 * m:
        indices = range(tot)
    else:
        indices = list(range(m)) + ['...'] + list(range(tot - m, tot))

    for i in indices:
        if i == '...':
            row = f"{'...':<10}" + "".join(f" {'...'.ljust(column_widths[key])}" for key in column_widths)
            print(row)
        else:
            print_row(i, data[i])