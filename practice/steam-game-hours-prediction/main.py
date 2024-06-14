import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = "drive/MyDrive/Colab Notebooks/kmu-rec-sys-24-hours/"


def readJSON(path):
  for l in open(path, 'r'):
    d = eval(l)
    u = d['userID']
    g = d['gameID']
    yield u,g,d

dataset = []
for uid, gid, d in readJSON(root + "train.json"):
  dataset.append((uid[1:], gid[1:], d))
dataset = np.array(dataset)

train_ratio = 0.9
validation_ratio = 0.1
# 데이터셋을 섞은 후 인덱스를 생성
indices = np.random.permutation(len(dataset))

#데이터셋을 분할하기 위한 인덱스 계산
train_idx = int(len(indices) * train_ratio)
val_idx = int(len(indices) * (train_ratio + validation_ratio))

# 학습용과 검증용 데이터셋으로 나누기
train_data = dataset[indices[:train_idx]]
val_data = dataset[indices[train_idx:val_idx]]

users = []
games = []
times = []
all_hours = []
rows = []
cols = []
user_hours = defaultdict(list)
game_mask = defaultdict(int)
user_mask = defaultdict(int)
comments = []
user_comments = []
for uid, gid, d in train_data:
  if not gid in game_mask:
    game_mask[gid] = len(game_mask)
    comments.append(0)
  if not uid in user_mask:
    user_mask[uid] = len(user_mask)
    user_comments.append(0)

  rows.append(user_mask[uid])
  cols.append(game_mask[gid])
  comments[game_mask[gid]] += 1
  user_comments[user_mask[uid]] += 1
  times.append(float(d['hours_transformed']))

users = np.array(rows)
games = np.array(cols)
times = np.array(times)

plt.figure(figsize=(10, 6))
plt.bar(list(game_mask.values()), comments, color='skyblue')
# plt.bar(list(game_mask.values()), times, color='skyblue')

# 그래프 제목 및 레이블 설정
plt.title('comments by game', fontsize=15)
plt.xlabel('game id', fontsize=12)
plt.ylabel('comments size', fontsize=12)

# 그래프 출력
plt.show()

times_tensor = torch.from_numpy(times).float().to(device)
users_tensor = torch.from_numpy(users).long().to(device)
games_tensor = torch.from_numpy(games).long().to(device)


alpha = torch.tensor(times.mean(), device=device, requires_grad=True)
user_bias = torch.randn(users.max() + 1, device=device, requires_grad=True)
game_bias = torch.randn(games.max() + 1, device=device, requires_grad=True)

optim = torch.optim.Adam([alpha, user_bias, game_bias], lr=0.5)

lmd = 0.01

for epoch in range(100):

  h = alpha + user_bias[users] + game_bias[games]
  mse = ((h - times_tensor) ** 2).mean()
  reg = lmd * ((game_bias ** 2).mean() + (user_bias ** 2).mean())
  cost = mse + reg

  optim.zero_grad()
  cost.backward()
  optim.step()

  with torch.no_grad():
    print((mse ** 0.5).item(), alpha.item())

test_iids = []
test_users = []
test_games = []

with open(root + "pairs_Hours.csv", "r") as f:
  print(f.readline())

  for line in f:
    iid, uid, gid = line.split(",")
    test_iids.append(iid)
    test_users.append(user_mask[int(uid[1:])])
    test_games.append(game_mask[int(gid[1:])])
test_users = np.array(test_users)
test_games = np.array(test_games)
test_users_tensor = torch.from_numpy(test_users).long().to(device)
test_games_tensor = torch.from_numpy(test_games).long().to(device)

with torch.no_grad():
    test_predictions = alpha + user_bias[test_users_tensor] + game_bias[test_games_tensor]

with open("hours_answer.csv", "w") as f2:
    f2.write("ID,Label\n") # insert the column names at the first row

    test_predictions_np = test_predictions.cpu().numpy()
    for i, pred in enumerate(test_predictions_np):
        if (i % 100) == 0:
            print(f'{i} User {test_users[i]}, Game {test_games[i]}, Predicted Play Time: {pred:.4f}')
        f2.write(f"{i},{pred:.4f}\n")
    f2.close()