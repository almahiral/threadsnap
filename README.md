# TweetSnap

Anda ingin membaca thread Twitter lebih cepat? Ayo gunakan TweetSnap!

TweetSnap adalah website yang dapat merangkum isi dari suatu thread Twitter dalam bahasa Indonesia.

TweetSnap menggunakan teknologi Transformer language model, lebih tepatnya mt5 [https://huggingface.co/docs/transformers/model_doc/mt5](https://huggingface.co/google/mt5-base)https://huggingface.co/google/mt5-base yang di fine-tune menggunakan dataset XLSum Indonesian [https://huggingface.co/datasets/csebuetnlp/xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum) untuk merangkum sebuah thread Twitter berbahasa Indonesia.

Adapun 2 metode pre-processing juga dilakukan, yaitu: 1) Semantic Search dan 2) Text Normalization. Kedua metode ini dilakukan untuk mengatasi bahasa sosial media yang sangat dinamis, tidak beraturan, dan mengandung bahasa informal dengan berbagai bentuk.
