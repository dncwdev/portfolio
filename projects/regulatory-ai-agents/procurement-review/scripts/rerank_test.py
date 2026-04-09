import requests

RERANKER_URL = "http://192.168.1.166:58002/rerank"

test_pairs = [
    {
        "query": "보안관리계획서 제출 요건",
        "passage": "계약자는 보안관리계획서를 제출하여야 한다"
    },
    {
        "query": "보안관리계획서 제출 요건",
        "passage": "산업안전보건법에 따른 안전교육 이수 의무"
    },
    {
        "query": "적용 법령",
        "passage": "국가를 당사자로 하는 계약에 관한 법률 시행령 제76조"
    }
]

for pair in test_pairs:
  response = requests.post(RERANKER_URL, json={
      "model": "/model",
      "query": pair["query"],
      "documents": [pair["passage"]]
  })
  score = response.json()["results"][0]["relevance_score"]
  print(f"score: {score:.4f} | {pair['passage'][:40]}")
