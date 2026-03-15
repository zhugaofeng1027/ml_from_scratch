import numpy as np

def adam_optimizer_for_linear_regression(X, y, lr=0.1, epochs=1000, beta1=0.9, beta2=0.99, eps=1e-8):
    w, b = 0.0, 0.0
    m_w, v_w = 0.0, 0.0
    m_b, v_b = 0.0, 0.0

    for t in range(1, epochs + 1):
        y_pred = w * X + b
        error = y_pred - y

        dw = np.mean(error * X)
        db = np.mean(error)

        m_w = beta1 * m_w + (1 - beta1) * dw
        m_b = beta1 * m_b + (1 - beta1) * db

        v_w = beta2 * v_w + (1 - beta2) * dw ** 2
        v_b = beta2 * v_b + (1 - beta2) * db ** 2
        
        m_w_hat = m_w / (1 - beta1 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    return w, b
   


        