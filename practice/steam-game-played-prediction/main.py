from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = "drive/MyDrive/Colab Notebooks/kmu-rec-sys-24-played/"

def readJSON(path):
  for l in open(path, 'r'):
    d = eval(l)
    u = d['userID']
    g = d['gameID']
    yield u,g,d

rows = []
cols = []
game_mask = defaultdict(int)
user_mask = defaultdict(int)
comments = []
user_comments = []
train_ratio = 0.9
validation_ratio = 0.1
dataset = []
for uid, gid, d in readJSON(root + "train.json"):
  dataset.append((uid, gid, d))
dataset = np.array(dataset)

# 데이터셋을 섞은 후 인덱스를 생성
indices = np.random.permutation(len(dataset))

#데이터셋을 분할하기 위한 인덱스 계산
train_idx = int(len(indices) * train_ratio)
val_idx = int(len(indices) * (train_ratio + validation_ratio))

# 학습용과 검증용 데이터셋으로 나누기
train_data = dataset[indices[:train_idx]]
val_data = dataset[indices[train_idx:val_idx]]

for uid, gid, d in train_data:
  _gid = gid[1:]
  _uid = uid[1:]
  if not _gid in game_mask:
    game_mask[_gid] = len(game_mask)
    comments.append(0)
  if not _uid in user_mask:
    user_mask[_uid] = len(user_mask)
    user_comments.append(0)
  rows.append(user_mask[_uid])
  cols.append(game_mask[_gid])
  comments[game_mask[_gid]] += 1
  user_comments[user_mask[_uid]] += 1

rows = np.array(rows)
cols = np.array(cols)

plt.figure(figsize=(10, 6))
plt.bar(list(game_mask.values()), comments, color='skyblue')

# 그래프 제목 및 레이블 설정
plt.title('comments by game', fontsize=15)
plt.xlabel('game id', fontsize=12)
plt.ylabel('comments size', fontsize=12)

# 그래프 출력
plt.show()

print(np.array(comments).mean())
print(np.median(np.array(comments)))

log_comments = np.log2(comments)
# normalized_comments = (log_comments - np.mean(log_comments)) / np.std(log_comments)

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(log_comments, bins=50, color='skyblue', edgecolor='black')
# 그래프 제목 및 레이블 설정
plt.title('comments by game', fontsize=15)
plt.xlabel('log2( amount of comments )', fontsize=12)
plt.ylabel('amount of games', fontsize=12)

# 그래프 출력
plt.show()

log_comments.mean()

log_user_comments = np.log2(user_comments)
normalized_user_comments = (log_user_comments - np.mean(log_user_comments)) / np.std(log_user_comments)
plt.figure(figsize=(10, 6))
plt.hist(normalized_user_comments, bins=50, color='skyblue', edgecolor='black')

# 그래프 제목 및 레이블 설정
plt.title('comments by user', fontsize=15)
plt.xlabel('user distribution', fontsize=12)
plt.ylabel('users size', fontsize=12)

# 그래프 출력
plt.show()

log_user_comments.mean()

users = torch.from_numpy(rows).long().to(device)
games = torch.from_numpy(cols).long().to(device)
# u_comments = torch.from_numpy(normalized_user_comments).float().to(device)
# g_comments = torch.from_numpy(normalized_comments).float().to(device)

n_factors = 10
n_games = games.max() + 1
n_users = users.max() + 1

game_bias = torch.randn(n_games, requires_grad=True, device=device)
game_factor = torch.randn(n_games, n_factors, requires_grad=True, device=device)
user_factor = torch.randn(n_users, n_factors, requires_grad=True, device=device)

len(log_comments)

lmd = 0.01
n_ratings = len(users)

optim = torch.optim.Adam([game_bias, game_factor, user_factor], lr=0.1)

logsigmoid = torch.nn.LogSigmoid()

for epoch in range(100):

  neg_games = torch.randint(1, n_games, (n_ratings,))

  pos_score = game_bias[games] + (user_factor[users] * game_factor[games]).sum(dim=1) #+ u_comments[users] + g_comments[games]
  neg_score = game_bias[neg_games] + (user_factor[users] * game_factor[neg_games]).sum(dim=1) #+ u_comments[users] + g_comments[games]
  reg = (game_bias ** 2).sum() + (game_factor ** 2).sum() + (user_factor ** 2).sum()

  cost = -logsigmoid(pos_score - neg_score).sum() + lmd * reg

  optim.zero_grad()
  cost.backward()
  optim.step()

  with torch.no_grad():

    train_acc = sum(pos_score > neg_score) / n_ratings
    print(f"epoch: {epoch}, train accuracy: {train_acc.item()}, cost: {cost.item()}")

train_acc = 0
results = []
for uid, gid, d in val_data:
  _gid = gid[1:]
  _uid = uid[1:]
  u = user_mask[_uid]
  g = game_mask[_gid]
  played_all = game_bias + (user_factor[u] * game_factor).sum(dim=1) #+ u_comments[u] + g_comments[g]
  result = played_all[g]
  piv = played_all.median()
  answer = 1 if result > piv else 0

  train_acc += answer
train_acc = train_acc / len(val_data)
print(f"acc: {train_acc:.4f}")

with torch.no_grad():

  n_outputs = 30
  uid = 2

  played_all = game_bias + (user_factor[uid] * game_factor).sum(dim=1)
  played_all_cpu = played_all.cpu().numpy()

  print(played_all.max(), played_all.min(), played_all.mean())
  print(played_all[39])

  ids = np.argsort(played_all_cpu)[::-1]
  pure_ids = ids[np.isin(ids, games[users == uid].cpu().numpy(), invert=True)]

  pure_ids = pure_ids[:n_outputs]
  played = played_all_cpu[pure_ids]

  print(pure_ids)
  print(played)

test_iids = []
test_users = []
test_games = []

with open(root + "pairs_Played.csv", "r") as f:
  print(f.readline())

  for line in f:
    iid, uid, gid = line.rstrip().split(",")
    test_iids.append(iid)
    test_users.append(user_mask[uid[1:]])
    test_games.append(game_mask[gid[1:]])

with open("played_answer.csv", "w") as f2:
    f2.write("ID,Label\n") # insert the column names at the first row
    for iid, uid, gid in zip(test_iids, test_users, test_games):
      played_all = game_bias + (user_factor[uid] * game_factor).sum(dim=1)# + u_comments[uid] + g_comments[gid]
      result = played_all[gid]
      piv = played_all.median()
      if (int(iid) % 100) == 0:
        print(f'{iid} User {uid}, Game {gid}, Predicted played: {1 if result > piv else 0}')
      f2.write(f"{iid},{1 if result > piv else 0}\n")
    f2.close()