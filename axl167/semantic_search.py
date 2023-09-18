# from: https://www.sbert.net/examples/applications/semantic-search/README.html

from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import re
import torch

embedder = SentenceTransformer('rahmanfadhil/indobert-finetuned-indonli')

# get the normalized text
text2 = "Kami mewawancarai sejarawan Minangkabau, sejarawan kuliner, guru besar ilmu gizi, dan 100 RM Padang untuk artikel ini: Semua yang harus kamu tahu tentang Nasi Padang.   Sebuah utas.  https://t.co/Fp6ggr1e3N  https://t.co/mo9Kf9DDj9. Kata ""Nasi Padang"", atau ""Rumah Makan Padang"" baru mulai populer pada akhir 1960-an, sebagai penamaan kontemporer bagi restoran Minangkabau. Sekarang, Nasi Padang ada di mana-mana, sampai di seluruh dunia. Lokasi RM Padang terjauh dari Padang ada di...  https://t.co/KlxN0Jasas. Yang jelas, survei BBC menemukan bahwa 79 persen warung Padang memberikan nasi lebih banyak untuk pesanan yang dibungkus.  https://t.co/cXpwlgF1eg  https://t.co/Ww8Azx5NLs. Menurut Guru Besar Ilmu Gizi Universitas Andalas, makan gorengan jauh lebih berbahaya dari makanan yang bersantan.   Selain itu, makanan Minang kaya akan bumbu seperti jahe, kunyit, lengkuas, dan daun serai yang berperan positif untuk tubuh. Bukan berarti boleh rakus ya.   Kami membuat kalkulator kalori untuk membantu menghitung berapa kalori yang terkandung pada sepiring Nasi Padang kamu.  https://t.co/cXpwlgF1eg  https://t.co/odyOrcQzKf. Misalnya, ternyata sepotong telur dadar berkalori jauh lebih tinggi daripada rendang.   Kalori terendah? Daun singkong 😅  https://t.co/cXpwlgF1eg  https://t.co/lVtPrfUPDD. Soal keaslian, sebagian besar warung Padang yang kami survei di Jabodetabek ternyata masih dimiliki oleh orang Minang. Sebagian masih pakai sistem bagi hasil.  https://t.co/yO4pe8W7rI. Baca artikel selengkapnya di tautan ini ya  https://t.co/cXpwlgF1eg  https://t.co/GnD52hoqqy."

# preprocess the text
def preprocess(text):
    # preprocess by remove non-alphabetical char because we don't need numerical values
    removed_num = re.sub(r"\b\d+\.", " ", text)

    # clean text after removal
    '''Remove whitespace before symbols and double periods'''
    clean_text = re.sub(r'\s+([?.!])', r'\1', removed_num)
    clean_text2 = re.sub(r"(\.)\1+", '.', clean_text)

    # split to sentences
    text_sentences = sent_tokenize(clean_text2)
    return text_sentences

def split2list(text_sentences):
    # split into 2 lists: main_sent and content_sent
    main = text_sentences[:5] # main contains 5 of first sentences, considered as main sentences
    content = text_sentences[5:] # content contains the rest of the sentences
    return main, content

def embedding_and_cos_sim(main, content):
# do embeddings from sentence transformers on main
    main_embeddings = embedder.encode(main, convert_to_tensor=True)

    # find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(main))

    score_dict = {}

    for query in content:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, main_embeddings)[0]
        top_results = torch.topk(cos_scores, k = top_k)
        total_score = 0
        for score, idx in zip(top_results[0], top_results[1]):
            total_score += score
        score_dict[query] = total_score
    return score_dict

def sentence_rank(score_dict, main, content):
    if (len(content) >= 50):
        top_n = 8
    elif (25 <= len(content) <= 49):
        top_n = 6
    elif (len(content) <= 24):
        top_n = 3

    bottom_scores = sorted(score_dict.items(), key=lambda item: item[1])[:top_n]

    # Get the queries with the bottom scores
    bottom_queries = [item[0] for item in bottom_scores]
    # print(bottom_queries)

    # Remove the bottom queries from the content list
    for query in bottom_queries:
        content.remove(query)

    main.extend(content)

    result = " ".join(main)
    return result, bottom_queries

def do_semantic_search(text):
    text_sentences = preprocess(text)
    main, content = split2list(text_sentences)
    score_dict = embedding_and_cos_sim(main, content)
    text_do_semantic_search = sentence_rank(score_dict, main, content)
    return text_do_semantic_search

# print(do_semantic_search(text2))