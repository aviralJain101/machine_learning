import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math


def predict_using_sklean():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_


def gradient_descent(x, y, learning_rate=0.0002, iterations=1000000):
    m_curr = b_curr = 0
    n=len(x)
    cost_previous=0
    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        cost = (1/n) * sum([val**2 for val in y-y_predicted])
        m_derivative = (-2/n) * sum(x*(y-y_predicted))
        b_derivative = (-2/n) * sum(y-y_predicted)

        m_curr = m_curr - learning_rate * m_derivative
        b_curr = b_curr - learning_rate * b_derivative

        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost

        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))
    
    return m_curr, b_curr


if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print()
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklean()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))
    print()

