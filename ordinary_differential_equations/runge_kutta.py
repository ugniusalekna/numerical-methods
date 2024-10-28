import numpy as np

from utils import timing, print_iteration_table
from plotting import show_plot


class ODESolver:
    def __init__(self, func, y0, t0=0, h=0.01, butcher_tableau=None, collect=True, true_solution=None):
        self.func = func
        self.h = h
        self.t = t0
        self.y0 = y0
        self.y_values = [y0]
        self.t_values = [t0]
        self.A, self.b, self.c = butcher_tableau
        self.collect = collect
        self.data = {}
        self.true_sol = true_solution
        
    def _update(self, y, t):
        self.y_values.append(y)
        self.t_values.append(t)

    def _runge_kutta_step(self, y, t, h):
        s = len(self.b)
        k = np.zeros((s, len(y)), dtype=float)
        
        for i in range(s):
            ti = t + self.c[i] * h
            yi = y + h * np.dot(self.A[i, :i], k[:i])
            k[i] = self.func(ti, yi)
        
        return y + h * np.dot(self.b, k)
    
    @timing
    def integrate(self, tf):
        y = self.y0

        if self.collect:
            entry = {'t': self.t, 'y': y}
            
            if self.true_sol:
                y_true = self.true_sol(self.t)
                err = np.linalg.norm(y - y_true, ord=2)
                entry.update({'y_true': y_true, 'error': err})
                
            self.data[0] = entry
            i = 1

        while self.t < tf:
            h = min(self.h, tf - self.t)
            y = self._runge_kutta_step(y, self.t, h)
            self.t = self.t + h
            self._update(y, self.t)
            
            if self.collect:
                entry = {'t': self.t, 'y': y}
                
                if self.true_sol:
                    y_true = self.true_sol(self.t)
                    err = np.linalg.norm(y - y_true, ord=2)
                    entry.update({'y_true': y_true, 'error': err})
                    
                self.data[i] = entry
                i += 1
        
        return np.array(self.y_values), np.array(self.t_values)
    
    def get_iteration_data(self):
        assert self.collect, "Iteration data not available unless collect=True."
        return self.data


def set_method(method):
    method = method.lower()
    
    if method == "rk1":  # Euler's forward
        A = np.array([[0]])
        b = np.array([1])
        c = np.array([0])
    
    elif method == "rk2":  # Midpoint
        A = np.array([[0, 0],
                      [0.5, 0]])
        b = np.array([0, 1])
        c = np.array([0, 0.5])
    
    elif method == "rk4":
        A = np.array([[0, 0, 0, 0],
                      [0.5, 0, 0, 0],
                      [0, 0.5, 0, 0],
                      [0, 0, 1, 0]])
        b = np.array([1/6, 1/3, 1/3, 1/6])
        c = np.array([0, 0.5, 0.5, 1])
    
    elif method == "rk45":
        A = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
        ])
        b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    
    else:
        raise ValueError(f"Method '{method}' not recognized.")

    return (A, b, c)


def solve_ivp(func, t_span, y0, method='rk1', h=0.001, return_data=False, solution=None):
    tableau = set_method(method)
    solver = ODESolver(func, y0=y0, t0=t_span[0], h=h, butcher_tableau=tableau, true_solution=solution)
    ys, ts = solver.integrate(tf=t_span[1])
    
    if return_data:
        data = solver.get_iteration_data()
        return (ys, ts), data
    
    return (ys, ts), None
        


def main():
    
    def func(t, y):
        return -y + np.sin(t)


    def solution(t):
        return 0.5 * (3 * np.exp(-t) + np.sin(t) - np.cos(t))


    t_span = [np.array([0]), np.array([5])]
    y0 = np.array([1])
    
    method = 'RK45'
    (ys, ts), data = solve_ivp(func, t_span, y0, method=method, h=0.001, return_data=True, solution=solution)
    print_iteration_table(data, m=5)
    
    show_plot(
        x_data=[ts, ts],
        y_data=[ys, solution(ts)],
        labels=[f'{method}', 'True solution'], 
        title='Solver vs true solution',
        x_label='t', 
        y_label='y(t)',
        line_styles=['-', '--'],
        colors=['blue', 'red'],
        markers=[None, None],
        figure_size=(9, 5)
    )


if __name__ == '__main__':
    main()