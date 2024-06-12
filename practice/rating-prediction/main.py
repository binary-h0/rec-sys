import numpy as np
import torch
from tqdm import tqdm

users = []
items = []
ratings = []

def gradient_descent_numpy():
    # Bias 모델
    alpha = ratings.mean()
    user_bias = np.zeros(users.max() + 1)
    items_bias = np.zeros(items.max() + 1)

    # Gradient Descent
    lr = 1
    lmd =  0.001

    n_ratings = len(ratings)
    n_users = len(user_bias)
    n_items = len(items_bias)

    for epoch in range(100):

        h = alpha + user_bias[users] + items_bias[items]
        diff = h - ratings

        rmse = (diff ** 2).mean() ** 0.5
        print(f"epoch: {epoch} rmse: {rmse}")

        grd_alpha = diff.mean()
        grd_user_bias = np.bincount(users, weights=diff) / n_ratings + lmd * user_bias / n_users
        grd_items_bias = np.bincount(items, weights=diff) / n_ratings + lmd * items_bias / n_items

        alpha = alpha - lr * grd_alpha
        user_bias = user_bias - lr * grd_user_bias
        items_bias = items_bias - lr * grd_items_bias
    
    h = alpha + user_bias[users] + items_bias[items]
    diff = h - ratings
    rmse = (diff ** 2).mean() ** 0.5
    print(f"rmse: {rmse}, alpha: {alpha}")

def gradient_descent_torch():
    
    ratings_tensor = torch.from_numpy(ratings)

    # Bias 모델
    alpha = torch.tensor(ratings.mean())
    alpha.requires_grad_(True)
    user_bias = torch.randn(users.max() + 1, requires_grad=True)
    item_bias = torch.randn(items.max() + 1, requires_grad=True)

    optimizer = torch.optim.Adam([alpha, user_bias, item_bias], lr=0.3)

    lmd = 0.001

    for epoch in range(100):
        
        h = alpha + user_bias[users] + item_bias[items]
        mse = ((h - ratings_tensor) ** 2).mean()
        reg = lmd * ((user_bias ** 2).mean() + (item_bias ** 2).mean())
        loss = mse + reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            print((mse ** 0.5).item(), alpha.item())

if __name__ == "__main__":
    root = "../data/ml-25m/"
    with open(root + "ratings.csv", "r") as f:
        print(f.readline())

        for line in f:
            uid, mid, rating, timestamp = line.strip().split(",")
            users.append(int(uid))
            items.append(int(mid))
            ratings.append(float(rating))

        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings)

    # 기본 모델의 RSME 계산
    print(((ratings - ratings.mean()) ** 2).mean() ** 0.5)

    # Numpy로 구현
    # gradient_descent_numpy()

    # Torch로 구현
    gradient_descent_torch()