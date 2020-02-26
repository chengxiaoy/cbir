from core.search import Search


search = Search(model, "../data/vectors.pkl", "../data/vectors.pkl", device, args=args)
paths, scores = search.search('../test/0a53f643515251.57f2991d49d25.jpg', 10)
print(scores)

