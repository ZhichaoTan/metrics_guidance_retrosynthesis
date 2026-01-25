"""ged_weight_change module.
"""

import numpy as np
import matplotlib.pyplot as plt

def constant_zero(x, start=0.05, end=0.05, max_iter=100):
    return 0

def constant(x, start=0.05, end=0.05, max_iter=100):
    return start

def linear_change(x, start=0, end=2, max_iter=100):
    return start + (end - start) * (x / max_iter)

def exponential_change(x, start=0, end=2, max_iter=100, epsilon=1e-8):
    ratio = (x + epsilon) / (max_iter + epsilon)
    return start + (end - start) * (np.expm1(ratio * np.log1p(max_iter))) / np.expm1(np.log1p(max_iter))

def logarithmic_change(x, start=0, end=2, max_iter=100, epsilon=1e-8):
    return start + (end - start) * np.log1p(x) / np.log1p(max_iter)

def sigmoid_change(x, start=0, end=2, max_iter=100, scale=10):
    return start + (end - start) / (1 + np.exp(- (x - max_iter / 2) / scale))

def hybrid_exponential_change(x, start=0, end=2, max_iter=100, epsilon=1e-8, slope_factor=0.25):
    start, end = start + epsilon, end + epsilon
    transition = min(int(max_iter/2), 50)
    # mid = start + (end - start) * (transition / max_iter) * slope_factor
    mid = 0.1
    alpha = np.log(end / mid) / (max_iter - transition)

    if x < transition:
        return start + (mid - start) * (x / transition)
    else:
        return mid * np.exp(alpha * (x - transition)) - epsilon

def hybrid_exponential_change_free(x, start=0, end=2, max_iter=100, epsilon=1e-8, slope_factor=0.25):
    start, end = start + epsilon, end + epsilon
    transition = int(max_iter/2)
    # mid = start + (end - start) * (transition / max_iter) * slope_factor
    mid = 0.1
    alpha = np.log(end / mid) / (max_iter - transition)

    if x < transition:
        return start + (mid - start) * (x / transition)
    else:
        return mid * np.exp(alpha * (x - transition)) - epsilon

ged_weight_change_fun_dict = {"constant_zero": constant_zero, "constant": constant, "linear_change": linear_change,
                              "exponential_change": exponential_change, "logarithmic_change": logarithmic_change,
                              "sigmoid_change": sigmoid_change, "hybrid_exponential_change": hybrid_exponential_change}

if __name__ == "__main__":
    max_iter = 100
    x_values = np.arange(0, max_iter+1, 1)
    # plt.plot(x_values, [constant(x) for x in x_values], label="Constant", linestyle="--")
    plt.plot(x_values, [linear_change(x, start=0, end=1, max_iter=max_iter) for x in x_values], label="Linear Change")
    plt.plot(x_values, [exponential_change(x, start=0, end=1, max_iter=max_iter) for x in x_values], label="Exponential Change")
    plt.plot(x_values, [logarithmic_change(x, start=0, end=1, max_iter=max_iter) for x in x_values], label="Logarithmic Change")
    plt.plot(x_values, [sigmoid_change(x, start=0, end=1, max_iter=max_iter) for x in x_values], label="Sigmoid Change")
    # plt.plot(x_values, [logarithmic_change(x, start=0, end=1, max_iter=max_iter) * sigmoid_change(x, start=0, end=1, max_iter=max_iter) for x in x_values], label="Log * hybrid")
    # plt.plot(x_values, [hybrid_exponential_change(x, start=0, end=1, max_iter=max_iter)/(1 - hybrid_exponential_change(x, start=0, end=1, max_iter=400) + 1e-8) for x in x_values], label="Hybrid Exponential Change_400")

    # for max_iter in [100, 200, 300, 400, 500]:
    #     # max_iter = 100
    #     x_values = np.arange(0, max_iter+1, 1)
    #     plt.plot(x_values, [hybrid_exponential_change(x, start=0, end=1, max_iter=max_iter) for x in x_values], label=f"Hybrid Exponential Change_{max_iter}")
    # plt.plot(x_values, [hybrid_exponential_change(x, start=0, end=1, max_iter=max_iter)/(1 - hybrid_exponential_change(x, start=0, end=1, max_iter=max_iter))  for x in x_values], label="Hybrid Exponential Change")

    plt.xlabel("Iteration (x)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("change_functions_increase.png")
    plt.show()

    import pandas as pd

    data = {
        "x": x_values,
        "Linear Change": [linear_change(x, start=0, end=1, max_iter=max_iter) for x in x_values],
        "Exponential Change": [exponential_change(x, start=0, end=1, max_iter=max_iter) for x in x_values],
        "Logarithmic Change": [logarithmic_change(x, start=0, end=1, max_iter=max_iter) for x in x_values],
        "Sigmoid Change": [sigmoid_change(x, start=0, end=1, max_iter=max_iter) for x in x_values]
    }

    df = pd.DataFrame(data)

    df.to_csv("change_results.csv", index=False)

    plt.plot(x_values, data["Linear Change"], label="Linear Change")
    plt.plot(x_values, data["Exponential Change"], label="Exponential Change")
    plt.plot(x_values, data["Logarithmic Change"], label="Logarithmic Change")
    plt.plot(x_values, data["Sigmoid Change"], label="Sigmoid Change")
    plt.legend()
    plt.show()

    # import pandas as pd
    # max_iters = [100, 200, 300, 400, 500]
    # data = {}

    # for max_iter in max_iters:
    #     x_values = np.arange(0, max_iter + 1)
    #     y_values = [hybrid_exponential_change(x, start=0, end=1, max_iter=max_iter) for x in x_values]
    #     data[max_iter] = pd.Series(y_values, index=x_values)

    # df = pd.DataFrame(data)
    # df.index.name = "x"

    # print(df)  # 打印输出
    # df.to_csv("hybrid_exponential_change_data.csv")