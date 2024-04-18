import csv
from collections import defaultdict
from tqdm import tqdm

movies = dict()
genres = dict()
ratings = []
watch_users = defaultdict(set)
user_ratings = defaultdict(dict)

def jaccard_similarity(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    if len(union) == 0: return 0

    return len(intersection) / len(union)

def cosine_similarity(vec1, vec2):
    numerator = sum(vec1[k] * vec2[k] for k in vec1.keys() & vec2.keys())
    denominator = (sum(v ** 2 for v in vec1.values()) * sum(v ** 2 for v in vec2.values())) ** 0.5
    if denominator == 0: return 0
    return numerator / denominator

def find_topk_jaccard_genres(target_mid, k=20):
    # TODO need integrity check for target_mid, k
    global movies, genres
    target_title = movies[target_mid]
    target_genres = genres[target_mid]
    res = []

    for mid, genres in genres.items():
        jaccard_score = jaccard_similarity(target_genres, genres)
        res.append( (jaccard_score, movies[mid]) )
    
    res.sort(reverse=True)
    return res[:k]

def find_topk_jaccard_ratings(target_mid, k=20):
    # TODO need integrity check for target_mid, k
    global ratings, watch_users
    target_watch_users = watch_users[target_mid]
    res = []

    for mid, uset in tqdm(watch_users.items()):
        if mid == target_mid: continue
        jaccard_score = jaccard_similarity(target_watch_users, uset)
        res.append( (jaccard_score, movies[mid]) )
    
    res.sort(reverse=True)
    return res[:k]

def find_topk_cosine_ratings(target_mid, k=20):
    # TODO need integrity check for target_mid, k
    global user_ratings, movies
    target_ratings = user_ratings[target_mid]
    res = []

    for mid, urset in tqdm(user_ratings.items()):
        if mid == target_mid: continue
        cosine_score = cosine_similarity(target_ratings, urset)
        res.append( (cosine_score, movies[mid]) )
    
    res.sort(reverse=True)
    return res[:k]


if __name__ == "__main__":
    with open("ml-25m/movies.csv") as f:
        csv_reader = csv.reader(f)
        next(csv_reader) # skip header

        for mid, title, genre in csv_reader:
            mid = int(mid)
            movies[mid] = title
            genres[mid] = set(genre.split("|"))

    
    with open("ml-25m/ratings.csv", "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for uid, mid, rating, timestamp in csv_reader:
            ratings.append((int(uid), int(mid), float(rating)))

    # 장르가 유사한 영화 찾기
    mid = 104841 # Gravity (2013)
    print("target:", movies[mid])
    res = find_topk_jaccard_genres(mid, 10)
    for score, title in res:
        print(f"{score * 100:.02f}% | {title}")

    # 다른 사용자가 함께 본 영화 찾기
    for uid, mid, rating in ratings:
        watch_users[mid].add(uid)
    mid = 104841 # Gravity (2013)
    print("target:", movies[mid])
    res = find_topk_jaccard_ratings(mid, 20)
    for score, title in res:
        print(f"%.02d{score * 100:.02f}% | {title}")

    # 비슷한 평가를 받는 영화 찾기
    for uid, mid, rating in ratings:
        user_ratings[uid][mid] = rating
    
    # 점수 보정 (Pearson Correlation Coefficient)
    for mid, urset in user_ratings.items():
        avg_rating = sum(urset.values()) / len(urset)
        for uid in urset:
            urset[uid] -= avg_rating
    mid = 104841 # Gravity (2013)
    print("target:", movies[mid])
    res = find_topk_cosine_ratings(mid, 20)
    for score, title in res:
        print(f"{score:.02f} | {title}")