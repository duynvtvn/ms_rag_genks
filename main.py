import random
import sys
import os
import json
import re
import logging
import torch
import torch.nn.functional as F
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset
import faiss
import nltk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModel,
    DPRContextEncoder,
    DPRQuestionEncoder,
    AdamW,
    get_constant_schedule
)
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedGENKSData(Dataset):
    """
    Lớp xử lý dữ liệu cho GENKS cải tiến, giải quyết vấn đề ánh xạ ID
    """

    def __init__(self, data, tokenizer, context_len=256, sent_len=64, max_length=1024, test=False, psg_filter=None,
                 psg_num=1, use_oracle=False, shuffle_id=False, max_id=128, add_aux_loss=False, gpt_style=False,
                 add_label=True, add_response=False, add_label_to_prefix=None, add_hyperlink=False,
                 use_pred_label=None, dialogue_first=True, knowledge_response=False, second_id=False, drop_null=True,
                 max_num_of_know=None):
        """
        Khởi tạo lớp xử lý dữ liệu GENKS cải tiến

        Args:
            data: Dữ liệu đầu vào
            tokenizer: Tokenizer để xử lý văn bản
            context_len: Độ dài tối đa của ngữ cảnh
            sent_len: Độ dài tối đa của mỗi câu
            max_length: Độ dài tối đa của chuỗi đầu vào
            test: Chế độ kiểm thử
            psg_filter: Bộ lọc đoạn văn
            psg_num: Số lượng đoạn văn sử dụng
            use_oracle: Sử dụng oracle hay không
            shuffle_id: Xáo trộn ID hay không
            max_id: ID tối đa
            add_aux_loss: Thêm loss phụ hay không
            gpt_style: Sử dụng kiểu GPT hay không
            add_label: Thêm nhãn hay không
            add_response: Thêm phản hồi hay không
            add_label_to_prefix: Thêm nhãn vào tiền tố
            add_hyperlink: Thêm hyperlink hay không
            use_pred_label: Sử dụng nhãn dự đoán
            dialogue_first: Đặt đối thoại lên đầu
            knowledge_response: Sử dụng tri thức làm phản hồi
            second_id: Sử dụng ID thứ hai
            drop_null: Bỏ qua mẫu null
            max_num_of_know: Số lượng tri thức tối đa
        """
        super(Dataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.sent_len = sent_len
        self.max_length = max_length
        self.test = test
        self.psg_filter = psg_filter
        self.psg_num = psg_num
        self.use_oracle = use_oracle
        self.shuffle_id = shuffle_id
        self.max_id = max_id
        self.add_aux_loss = add_aux_loss
        self.gpt_style = gpt_style
        self.add_response = add_response
        self.add_label = add_label
        self.response = [example.get('labels', [''])[0] for example in self.data]
        self.add_label_to_prefix = add_label_to_prefix
        self.add_hyperlink = add_hyperlink
        self.use_pred_label = use_pred_label
        self.dialogue_first = dialogue_first
        self.knowledge_response = knowledge_response
        self.second_id = second_id
        self.drop_null = drop_null
        self.max_num_of_know = max_num_of_know
        self._id_map = None
        self._sentence_to_id = {}  # Lưu ánh xạ từ câu đến ID
        self._debug_info = {}  # Lưu thông tin debug

    def get_id_map(self):
        """
        Trả về ánh xạ ID - quan trọng cho việc giải mã định danh tri thức

        Returns:
            List ID đã ánh xạ
        """
        if self._id_map is None:
            # Tạo ánh xạ ID mới
            id_map = [i for i in range(2, self.max_id)]
            if self.shuffle_id:
                np.random.shuffle(id_map)
            self._id_map = [0, 1] + id_map
        return self._id_map

    def get_sentence_to_id_map(self):
        """
        Trả về ánh xạ từ câu đến ID

        Returns:
            Dict ánh xạ từ câu đến ID
        """
        return self._sentence_to_id

    def get_debug_info(self):
        """
        Trả về thông tin debug

        Returns:
            Dict thông tin debug
        """
        return self._debug_info

    def __getitem__(self, index):
        """
        Lấy mẫu dữ liệu tại vị trí index, có ghi lại ánh xạ ID

        Args:
            index: Vị trí mẫu dữ liệu

        Returns:
            Tuple (input_ids, labels)
        """
        example = self.data[index]
        id_map = self.get_id_map()

        # =============================
        # Xử lý tri thức
        # =============================
        knowledge = example['knowledge']
        titles = list(knowledge.keys())[:self.psg_num]

        if self.use_oracle and example.get('title', '') != 'no_passages_used' and \
                example.get('title', '') in knowledge and example.get('title', '') not in titles:
            titles = [example.get('title', '')] + titles[:-1]

        new_knowledge = OrderedDict()
        for k in titles:
            new_knowledge[k] = knowledge[k]
        knowledge = new_knowledge

        if self.drop_null and not self.test and example.get('title', '') != 'no_passages_used':
            if example.get('title', '') not in knowledge or \
                    example.get('checked_sentence', '') not in knowledge[example.get('title', '')]:
                return self[np.random.randint(len(self))]

        # =============================
        # Tạo chuỗi tri thức
        # =============================
        knowledge_sequence = []
        dialogue_sequence = []
        sentence_to_id = {}
        sent_id = 0
        label = f'<s{id_map[0]}>'  # Mặc định là nhãn đầu tiên

        # Phần thông tin tri thức
        knowledge_sequence += self.tokenizer.encode('\nPassage information.\n', add_special_tokens=False)

        # Mục mặc định "no_passages_used"
        sentence = 'no_passages_used'
        sent_id += 1
        knowledge_sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>{sentence}<s{id_map[sent_id]}>\n',
                                                    add_special_tokens=False)
        sentence_to_id[sentence] = sent_id
        if 'checked_sentence' in example and sentence == example.get('checked_sentence', ''):
            label = f'<s{id_map[sent_id]}>'

        # Thêm các tri thức từ knowledge
        for pid, (title, passage) in enumerate(knowledge.items()):
            knowledge_sequence += self.tokenizer.encode(f'Passage {pid + 1}, Title: {title}\n',
                                                        add_special_tokens=False)
            for sentence in passage:
                if len(knowledge_sequence) > self.max_length:
                    break
                sent_id += 1
                knowledge_sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>{sentence}',
                                                            truncation=True, max_length=self.sent_len,
                                                            add_special_tokens=False)
                knowledge_sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>\n', add_special_tokens=False)
                sentence_to_id[sentence] = sent_id
                if 'checked_sentence' in example and sentence == example.get('checked_sentence', ''):
                    label = f'<s{id_map[sent_id]}>'

                if self.max_num_of_know is not None and sent_id >= self.max_num_of_know:
                    break
            if self.max_num_of_know is not None and sent_id >= self.max_num_of_know:
                break

        # Lưu ánh xạ từ câu đến ID để sử dụng sau này
        self._sentence_to_id = sentence_to_id.copy()

        # Lưu thông tin debug
        self._debug_info = {
            'example_id': index,
            'label': label,
            'id_map': id_map.copy(),
            'sentence_to_id': sentence_to_id.copy(),
            'checked_sentence': example.get('checked_sentence', ''),
            'title': example.get('title', '')
        }

        # =============================
        # Tạo chuỗi đối thoại
        # =============================
        role = {'0_Wizard': 'User1: ', '1_Apprentice': 'User2: ', '0_Apprentice': 'User2: ', '1_Wizard': 'User1: ',
                0: 'User1: ', 1: 'User2: ', 'user1': 'User1: ', 'user2': 'User2: '}
        context = ''

        # Xây dựng chuỗi đối thoại từ lịch sử
        for turn in example.get('context', []):
            speaker = role.get(turn.get('speaker', ''), turn.get('speaker', ''))
            text = turn.get('text', '')
            kk = ''
            if self.add_hyperlink and 'title' in turn and 'checked_sentence' in turn:
                kk = f"[{turn['title']}]"
                if turn['checked_sentence'] in sentence_to_id:
                    kk += f"<s{id_map[sentence_to_id[turn['checked_sentence']]]}>"
                kk += ' '
            context += f'{speaker}{kk}{text}\n'

        # Thêm thông tin chủ đề và ngữ cảnh
        topic = 'Chosen topic: ' + example.get('chosen_topic', '') + '\n'
        dialogue_sequence += self.tokenizer.encode('\nDialogue history.\n', add_special_tokens=False)
        dialogue_sequence += self.tokenizer.encode(topic, add_special_tokens=False)
        dialogue_sequence += self.tokenizer.encode(context, add_special_tokens=False)[-self.context_len:]
        dialogue_sequence += self.tokenizer.encode('Predict the next knowledge sentence id and response of User1.\n',
                                                   add_special_tokens=False)

        # Thêm nhãn vào tiền tố nếu cần
        if self.add_label_to_prefix:
            if isinstance(self.add_label_to_prefix, list):
                pred_label = self.add_label_to_prefix[index]
                dialogue_sequence += self.tokenizer.encode(f'Selected knowledge = {pred_label}\n',
                                                           add_special_tokens=False)
            else:
                dialogue_sequence += self.tokenizer.encode(f'Selected knowledge = {label}\n',
                                                           add_special_tokens=False)

        # =============================
        # Kết hợp các chuỗi đầu vào/đầu ra
        # =============================
        sequence = []
        knowledge_sequence = knowledge_sequence[:self.max_length - len(dialogue_sequence)]
        if self.dialogue_first:
            sequence += dialogue_sequence
            sequence += knowledge_sequence
        else:
            sequence += knowledge_sequence
            sequence += dialogue_sequence

        # Xây dựng đầu ra
        target = []
        if self.add_label:
            if isinstance(self.use_pred_label, list):
                target.append(self.use_pred_label[index][0])
            else:
                target.append(f'{label}')
        if self.add_response:
            if self.knowledge_response and example.get('checked_sentence', '') != 'no_passages_used' and \
                    np.random.random() < self.knowledge_response:
                target.append(f"{example.get('checked_sentence', '')}")
            else:
                target.append(f"{example.get('labels', [''])[0]}")
        target = ' '.join(target)

        # Định dạng đầu vào và đầu ra theo kiểu seq2seq (BART)
        sequence = sequence
        labels = self.tokenizer.encode(target, truncation=True, max_length=self.context_len,
                                       add_special_tokens=True)

        return torch.tensor(sequence), torch.tensor(labels)

    def __len__(self):
        """
        Trả về số lượng mẫu dữ liệu

        Returns:
            Số lượng mẫu
        """
        return len(self.data)

    def collate_fn(self, data):
        """
        Hàm gộp batch cho DataLoader

        Args:
            data: Danh sách các mẫu

        Returns:
            Dict batch đã gộp
        """
        from torch.nn.utils.rnn import pad_sequence
        padding_value = self.tokenizer.pad_token_id
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(padding_value),
            'labels': labels,
        }


class ImprovedMultiStageRAGWithGENKS:
    """
    Hệ thống RAG đa giai đoạn kết hợp với GENKS đã được cải tiến
    Quy trình:
    1. Truy xuất sơ bộ: Sử dụng DPR hoặc SBERT để lấy tập lớn đoạn văn liên quan
    2. Lọc và xếp hạng: Sử dụng mô hình xếp hạng để chọn đoạn văn liên quan nhất
    3. GENKS: Sử dụng mô hình GENKS để sinh định danh tri thức với cải tiến ánh xạ ID
    4. Sinh phản hồi: Sinh phản hồi dựa trên tri thức đã chọn
    """

    def __init__(self,
                 model_name='facebook/bart-base',
                 retriever_model_name='facebook/dpr-ctx_encoder-single-nq-base',
                 query_encoder_name='facebook/dpr-question_encoder-single-nq-base',
                 ranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 embed_dim=768,
                 top_k_retrieval=20,
                 top_k_rerank=5,
                 retrieval_method='bm25',
                 cache_dir=None):
        """
        Khởi tạo hệ thống RAG đa giai đoạn cải tiến

        Args:
            model_name: Tên mô hình GENKS
            retriever_model_name: Tên mô hình bộ truy xuất
            query_encoder_name: Tên mô hình mã hóa truy vấn
            ranker_model_name: Tên mô hình xếp hạng lại
            embed_dim: Chiều của embedding
            top_k_retrieval: Số lượng tài liệu truy xuất
            top_k_rerank: Số lượng tài liệu sau khi xếp hạng lại
            retrieval_method: Phương pháp truy xuất ('bm25' hoặc 'dpr')
            cache_dir: Thư mục cache
        """
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.retrieval_method = retrieval_method.lower()

        # Giai đoạn 1: Bộ truy xuất sơ bộ
        self.initialize_retrievers(retriever_model_name, query_encoder_name, embed_dim)

        # Giai đoạn 2: Bộ xếp hạng lại
        self.initialize_ranker(ranker_model_name)

        # Giai đoạn 3 & 4: GENKS & Sinh phản hồi
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Thêm tokens đặc biệt cho GENKS
        self.tokenizer.add_tokens(
            [f'<s{i}>' for i in range(128)] + ['<s>', '</s>', '<pad>', '<positive>', '<negative>'])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Bộ nhớ đệm tài liệu đã truy xuất
        self.doc_cache = {}

        # Cache cho ánh xạ ID
        self.id_cache = {}

        # Cache cho sentence_to_id
        self.sentence_id_cache = {}

        # Thông tin debug
        self.debug_info = {}

    def initialize_retrievers(self, retriever_model_name, query_encoder_name, embed_dim):
        """
        Khởi tạo các bộ truy xuất: DPR hoặc SBERT + FAISS và BM25

        Args:
            retriever_model_name: Tên mô hình bộ truy xuất
            query_encoder_name: Tên mô hình mã hóa truy vấn
            embed_dim: Chiều của embedding
        """
        # BM25 luôn được khởi tạo để sẵn sàng sử dụng
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_class = BM25Okapi
            self.bm25 = None  # Sẽ được khởi tạo trong build_corpus_index
            self.tokenized_corpus = None
            logger.info("Đã khởi tạo BM25")
        except ImportError:
            logger.warning("Không thể nhập rank_bm25. Cài đặt bằng 'pip install rank-bm25'")
            self.bm25_class = None

        # Nếu method là 'dpr' hoặc DPR có sẵn, khởi tạo DPR
        if self.retrieval_method != 'bm25':
            try:
                self.ctx_encoder = DPRContextEncoder.from_pretrained(retriever_model_name)
                self.query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder_name)
                self.dpr_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
                self.query_tokenizer = AutoTokenizer.from_pretrained(query_encoder_name)
                logger.info("Đã khởi tạo bộ truy xuất DPR")
            except Exception as e:
                logger.info(f"Không thể khởi tạo DPR: {str(e)}, sẽ sử dụng phương pháp thay thế")
                if self.retrieval_method == 'dpr':
                    # Fallback sang SentenceTransformer nếu yêu cầu DPR nhưng không thể tải
                    self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                    self.index = faiss.IndexFlatIP(embed_dim)
                    logger.info("Đã chuyển sang SentenceTransformer")

        # Luôn khởi tạo TF-IDF như backup cuối cùng
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def initialize_ranker(self, ranker_model_name):
        """
        Khởi tạo bộ xếp hạng lại dựa trên cross-encoder

        Args:
            ranker_model_name: Tên mô hình cross-encoder
        """
        try:
            from sentence_transformers import CrossEncoder
            self.ranker = CrossEncoder(ranker_model_name)
            logger.info(f"Đã khởi tạo bộ xếp hạng CrossEncoder: {ranker_model_name}")
        except:
            logger.info("Không thể khởi tạo CrossEncoder, sẽ sử dụng phương pháp xếp hạng đơn giản")
            self.ranker = None

    def build_corpus_index(self, corpus, cache_path=None):
        """
        Xây dựng index cho corpus để truy xuất nhanh với cả DPR và BM25

        Args:
            corpus: Danh sách các văn bản [(id, text, title)]
            cache_path: Đường dẫn để lưu/tải embeddings từ cache
        """
        self.corpus = corpus
        self.id_to_doc = {item[0]: (item[1], item[2]) for item in corpus}
        texts = [doc[1] for doc in corpus]

        # Luôn xây dựng BM25 vì nó nhẹ và nhanh
        if hasattr(self, 'bm25_class') and self.bm25_class is not None:
            logger.info("Đang xây dựng BM25 index...")
            # Tokenize corpus cho BM25
            self.tokenized_corpus = [text.lower().split() for text in texts]
            self.bm25 = self.bm25_class(self.tokenized_corpus)
            logger.info("Đã xây dựng BM25 index thành công")

        # Nếu phương thức truy xuất không phải BM25, tiếp tục với DPR hoặc phương thức khác
        if self.retrieval_method != 'bm25':
            # Tải embedding từ cache nếu có
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Đang tải embeddings từ cache: {cache_path}")
                self.corpus_embeddings = torch.load(cache_path)
                # Xây dựng FAISS index
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.size(1))
                self.index.add(self.corpus_embeddings.cpu().numpy())
                logger.info("Đã tải và xây dựng index từ cache thành công")
                return

            # Xây dựng index với DPR hoặc SentenceTransformer
            if hasattr(self, 'ctx_encoder') and hasattr(self, 'dpr_tokenizer'):
                # Sử dụng DPR để xây dựng index
                logger.info("Đang tạo embeddings cho corpus với DPR...")
                self.corpus_embeddings = []

                batch_size = 32
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = texts[i:i + batch_size]
                    with torch.no_grad():
                        inputs = self.dpr_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True,
                                                    max_length=512)
                        embeddings = self.ctx_encoder(**inputs).pooler_output
                    self.corpus_embeddings.append(embeddings)

                self.corpus_embeddings = torch.cat(self.corpus_embeddings, dim=0)

                # Lưu embeddings vào cache nếu có đường dẫn
                if cache_path:
                    logger.info(f"Đang lưu embeddings vào cache: {cache_path}")
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(self.corpus_embeddings, cache_path)

                # Xây dựng FAISS index
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
                self.index.add(self.corpus_embeddings.cpu().numpy())

            elif hasattr(self, 'encoder'):
                # Sử dụng Sentence-Transformer
                logger.info("Đang tạo embeddings cho corpus với SentenceTransformer...")
                texts = [doc[1] for doc in corpus]
                self.corpus_embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_tensor=True)
                # Xây dựng FAISS index
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
                self.index.add(self.corpus_embeddings.cpu().numpy())
            else:
                # Fallback sang TF-IDF
                logger.info("Đang tạo TF-IDF matrix...")
                texts = [doc[1] for doc in corpus]
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def retrieve_documents(self, query, dialogue_history=None, top_k=None):
        """
        Giai đoạn 1: Truy xuất tài liệu sơ bộ, hỗ trợ cả DPR và BM25

        Args:
            query: Câu truy vấn
            dialogue_history: Lịch sử đối thoại (tùy chọn)
            top_k: Số lượng tài liệu cần truy xuất

        Returns:
            List of retrieved documents [(id, text, title, score)]
        """
        if top_k is None:
            top_k = self.top_k_retrieval

        # Kết hợp truy vấn với lịch sử đối thoại nếu có
        if dialogue_history:
            last_n_turns = dialogue_history[-3:]  # Lấy 3 lượt cuối
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        # Kiểm tra cache
        if combined_query in self.doc_cache:
            logger.info(f"Sử dụng kết quả từ bộ nhớ đệm cho truy vấn: {combined_query[:50]}...")
            return self.doc_cache[combined_query]

        # Sử dụng BM25 nếu được chọn và đã khởi tạo
        if self.retrieval_method == 'bm25' and hasattr(self, 'bm25') and self.bm25 is not None:
            logger.info("Đang truy xuất với BM25...")
            # Tokenize truy vấn giống cách tokenize corpus
            tokenized_query = combined_query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            indices = np.argsort(-scores)[:top_k]

            # Chuyển indices thành documents
            retrieved_docs = []
            for i, idx in enumerate(indices):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(scores[idx])))

        # Sử dụng DPR hoặc phương pháp thay thế khác
        elif hasattr(self, 'query_encoder'):
            # Sử dụng DPR
            with torch.no_grad():
                inputs = self.query_tokenizer(combined_query, return_tensors="pt")
                query_embedding = self.query_encoder(**inputs).pooler_output.cpu().numpy()

            # Tìm kiếm với FAISS
            scores, indices = self.index.search(query_embedding, top_k)

            # Chuyển indices thành documents
            retrieved_docs = []
            for i, (idx, score) in enumerate(zip(indices.flatten(), scores.flatten())):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        elif hasattr(self, 'encoder'):
            # Sử dụng Sentence-Transformer
            query_embedding = self.encoder.encode(combined_query, convert_to_tensor=True)
            # Tìm kiếm với FAISS
            scores, indices = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), top_k)

            # Chuyển indices thành documents
            retrieved_docs = []
            for i, (idx, score) in enumerate(zip(indices.flatten(), scores.flatten())):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        else:
            # Fallback sang TF-IDF
            query_vec = self.tfidf_vectorizer.transform([combined_query])
            scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
            indices = np.argsort(-scores)[:top_k]
            scores = scores[indices]

            # Chuyển indices thành documents
            retrieved_docs = []
            for i, (idx, score) in enumerate(zip(indices.flatten(), scores.flatten())):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        # Lưu vào cache
        self.doc_cache[combined_query] = retrieved_docs

        return retrieved_docs

    def rerank_documents(self, query, docs, dialogue_history=None, top_k=None):
        """
        Giai đoạn 2: Lọc và xếp hạng lại tài liệu

        Args:
            query: Câu truy vấn
            docs: Danh sách tài liệu từ giai đoạn 1
            dialogue_history: Lịch sử đối thoại (tùy chọn)
            top_k: Số lượng tài liệu cần trả về

        Returns:
            List of reranked documents [(id, text, title, score)]
        """
        if top_k is None:
            top_k = self.top_k_rerank

        # Kết hợp truy vấn với lịch sử đối thoại nếu có
        if dialogue_history:
            # Lấy n câu cuối từ lịch sử đối thoại
            last_n_turns = dialogue_history[-3:]  # Lấy 3 lượt cuối
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        if self.ranker:
            # Sử dụng cross-encoder cho việc xếp hạng
            pairs = [(combined_query, doc[1]) for doc in docs]
            scores = self.ranker.predict(pairs)

            # Xếp hạng lại dựa trên điểm số mới
            reranked_docs = [(docs[i][0], docs[i][1], docs[i][2], float(scores[i]))
                             for i in range(len(docs))]
            reranked_docs = sorted(reranked_docs, key=lambda x: x[3], reverse=True)[:top_k]
        else:
            # Đơn giản sắp xếp lại dựa trên điểm truy xuất gốc nếu không có ranker
            reranked_docs = sorted(docs, key=lambda x: x[3], reverse=True)[:top_k]

        return reranked_docs

    def prepare_genks_input(self, query, docs, dialogue_history=None):
        """
        Chuẩn bị đầu vào cho GENKS từ các tài liệu đã xếp hạng

        Args:
            query: Câu truy vấn
            docs: Danh sách tài liệu từ giai đoạn 2
            dialogue_history: Lịch sử đối thoại

        Returns:
            Dữ liệu đầu vào cho GENKS
        """
        # Xây dựng lịch sử đối thoại
        context = []
        if dialogue_history:
            for i, utterance in enumerate(dialogue_history):
                speaker = "User1: " if i % 2 == 0 else "User2: "
                context.append({"speaker": speaker.strip(':'), "text": utterance})

        # Thêm truy vấn hiện tại
        context.append({"speaker": "User2: " if len(context) % 2 == 0 else "User1: ", "text": query})

        # Chuyển đổi tài liệu thành định dạng GENKS
        knowledge = OrderedDict()
        for doc_id, doc_text, doc_title, _ in docs:
            # Phân đoạn tài liệu thành các câu
            try:
                sentences = nltk.sent_tokenize(doc_text)
            except:
                # Fallback nếu không thể tokenize
                sentences = [doc_text]

            if doc_title not in knowledge:
                knowledge[doc_title] = []
            knowledge[doc_title].extend(sentences)

        # Xây dựng đầu vào GENKS
        genks_input = {
            "knowledge": knowledge,
            "context": context,
            "chosen_topic": docs[0][2] if docs else "Unknown",  # Sử dụng tiêu đề tài liệu đầu tiên làm chủ đề
            "title": "",  # Sẽ được điền sau khi chọn tri thức
            "checked_sentence": "",  # Sẽ được điền sau khi chọn tri thức
            "labels": [""]  # Sẽ được điền sau khi sinh phản hồi
        }

        return genks_input

    def select_knowledge(self, genks_input, device='cuda'):
        """
        Giai đoạn 3: Sử dụng GENKS để chọn tri thức với cải tiến ánh xạ ID

        Args:
            genks_input: Dữ liệu đầu vào cho GENKS
            device: Thiết bị tính toán

        Returns:
            Định danh tri thức đã chọn, tiêu đề và nội dung tri thức tương ứng
        """
        # Chuyển sang thiết bị tính toán
        self.model = self.model.to(device)

        # Chuẩn bị dữ liệu GENKS với lớp xử lý dữ liệu cải tiến
        dataset = ImprovedGENKSData([genks_input], self.tokenizer, context_len=128, sent_len=64, max_length=512,
                                    psg_num=len(genks_input["knowledge"]), shuffle_id=False, max_id=128,
                                    add_aux_loss=False, gpt_style=False, use_oracle=False,
                                    add_label=True, add_response=False, add_hyperlink=True,
                                    dialogue_first=True, test=True)

        # Lưu ánh xạ ID và sentence_to_id vào cache
        example_id = hash(str(genks_input))
        self.id_cache[example_id] = dataset.get_id_map()
        self.sentence_id_cache[example_id] = dataset.get_sentence_to_id_map()
        self.debug_info[example_id] = dataset.get_debug_info()

        # Lấy dữ liệu đầu vào
        input_ids, _ = dataset[0]
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Sinh định danh tri thức
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=16,  # Tăng max_length để đảm bảo ID được sinh đầy đủ
                num_beams=4
            )

        # Giải mã đầu ra - không bỏ qua special tokens
        knowledge_id_raw = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Trích xuất ID chính xác từ đầu ra
        knowledge_id = self._extract_knowledge_id(knowledge_id_raw)

        # Ánh xạ định danh tri thức về đoạn tri thức tương ứng
        selected_title = None
        selected_sentence = None

        # Lấy ánh xạ từ cache
        id_map = self.id_cache[example_id]
        sentence_to_id = self.sentence_id_cache[example_id]

        # Phân tích định danh để tìm câu tương ứng
        for title, sentences in genks_input["knowledge"].items():
            for sentence in sentences:
                if sentence in sentence_to_id:
                    sent_id = sentence_to_id[sentence]
                    sent_tag = f'<s{id_map[sent_id]}>'

                    # So sánh ID với độ linh hoạt cao hơn
                    if self._is_id_match(sent_tag, knowledge_id):
                        selected_title = title
                        selected_sentence = sentence
                        break
            if selected_title:
                break

        # Nếu không tìm thấy, sử dụng đoạn đầu tiên
        if not selected_title and genks_input["knowledge"]:
            selected_title = list(genks_input["knowledge"].keys())[0]
            selected_sentence = genks_input["knowledge"][selected_title][0]

        # Lưu thông tin cho debug
        self.debug_info[example_id].update({
            'generated_id_raw': knowledge_id_raw,
            'generated_id': knowledge_id,
            'selected_title': selected_title,
            'selected_sentence': selected_sentence
        })

        return knowledge_id, selected_title, selected_sentence

    def _extract_knowledge_id(self, text):
        """
        Trích xuất ID tri thức từ văn bản theo định dạng <sX>

        Args:
            text: Văn bản chứa ID

        Returns:
            ID tri thức đã trích xuất
        """
        pattern = r'<s\d+>'
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
        return text

    def _is_id_match(self, id1, id2):
        """
        So sánh ID với độ linh hoạt cao hơn

        Args:
            id1: ID thứ nhất
            id2: ID thứ hai

        Returns:
            True nếu hai ID khớp nhau, False nếu không
        """
        # Chuẩn hóa định dạng ID
        id1 = id1.strip().replace(' ', '')
        id2 = id2.strip().replace(' ', '')

        # So sánh cơ bản
        if id1 == id2:
            return True

        # So sánh số ID (bỏ qua các ký tự định dạng)
        try:
            num1 = int(re.search(r'<s(\d+)>', id1).group(1))
            num2 = int(re.search(r'<s(\d+)>', id2).group(1))
            return num1 == num2
        except:
            return False

    def generate_response(self, genks_input, selected_title, selected_sentence, device='cuda'):
        """
        Giai đoạn 4: Sinh phản hồi dựa trên tri thức đã chọn

        Args:
            genks_input: Dữ liệu đầu vào cho GENKS
            selected_title: Tiêu đề tri thức đã chọn
            selected_sentence: Câu tri thức đã chọn
            device: Thiết bị tính toán

        Returns:
            Phản hồi được sinh ra
        """
        # Cập nhật thông tin tri thức đã chọn
        genks_input["title"] = selected_title
        genks_input["checked_sentence"] = selected_sentence

        # Chuẩn bị dữ liệu GENKS với cờ add_response=True
        dataset = ImprovedGENKSData([genks_input], self.tokenizer, context_len=128, sent_len=64, max_length=512,
                                    psg_num=len(genks_input["knowledge"]), shuffle_id=False, max_id=128,
                                    add_aux_loss=False, gpt_style=False, use_oracle=False, test=True,
                                    add_label=False, add_response=True, add_hyperlink=True,
                                    add_label_to_prefix=True, dialogue_first=True)

        # Lấy dữ liệu đầu vào
        input_ids, _ = dataset[0]
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Sinh phản hồi
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=3,
                temperature=0.7
            )

        # Giải mã đầu ra
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def process_query(self, query, dialogue_history=None, device='cuda'):
        """
        Xử lý truy vấn theo quy trình đa giai đoạn đầy đủ

        Args:
            query: Câu truy vấn
            dialogue_history: Lịch sử đối thoại
            device: Thiết bị tính toán

        Returns:
            Phản hồi cuối cùng và thông tin trung gian
        """
        # Giai đoạn 1: Truy xuất tài liệu sơ bộ
        retrieved_docs = self.retrieve_documents(query, dialogue_history)
        logger.info(f"Giai đoạn 1: Đã truy xuất {len(retrieved_docs)} tài liệu")

        # Giai đoạn 2: Lọc và xếp hạng lại
        reranked_docs = self.rerank_documents(query, retrieved_docs, dialogue_history)
        logger.info(f"Giai đoạn 2: Đã xếp hạng lại lấy {len(reranked_docs)} tài liệu tốt nhất")

        # Chuẩn bị đầu vào cho GENKS
        genks_input = self.prepare_genks_input(query, reranked_docs, dialogue_history)

        # Giai đoạn 3: Chọn tri thức với GENKS
        knowledge_id, selected_title, selected_sentence = self.select_knowledge(genks_input, device)
        logger.info(f"Giai đoạn 3: Đã chọn tri thức từ '{selected_title}': {selected_sentence[:50]}...")

        # Giai đoạn 4: Sinh phản hồi
        response = self.generate_response(genks_input, selected_title, selected_sentence, device)
        logger.info(f"Giai đoạn 4: Đã sinh phản hồi: {response[:50]}...")

        # Lấy thông tin debug
        example_id = hash(str(genks_input))
        debug_data = self.debug_info.get(example_id, {})

        return {
            "response": response,
            "selected_knowledge": {
                "title": selected_title,
                "sentence": selected_sentence,
                "knowledge_id": knowledge_id
            },
            "retrieved_docs": retrieved_docs[:5],  # Chỉ trả về 5 tài liệu đầu tiên để giảm kích thước
            "reranked_docs": reranked_docs[:3],
            "debug_info": debug_data  # Thông tin debug
        }

    def save(self, path):
        """
        Lưu mô hình

        Args:
            path: Đường dẫn thư mục lưu mô hình
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(os.path.join(path, "genks_model"))
        self.tokenizer.save_pretrained(os.path.join(path, "genks_tokenizer"))

    def load(self, path, device='cuda'):
        """
        Tải mô hình

        Args:
            path: Đường dẫn thư mục chứa mô hình
            device: Thiết bị tính toán
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, "genks_model")).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "genks_tokenizer"))


def improved_evaluate_multi_stage_rag_genks(model, eval_data, output_file=None, batch_size=16):
    """
    Đánh giá cải tiến cho mô hình RAG đa giai đoạn kết hợp GENKS

    Args:
        model: Mô hình ImprovedMultiStageRAGWithGENKS
        eval_data: Dữ liệu đánh giá
        output_file: File lưu kết quả đánh giá
        batch_size: Kích thước batch

    Returns:
        Từ điển kết quả đánh giá
    """
    logger.info(f"Đang đánh giá mô hình với {len(eval_data)} mẫu")

    # Chuẩn bị dữ liệu đánh giá
    eval_dataset = ImprovedGENKSData(
        eval_data,
        model.tokenizer,
        context_len=128,
        sent_len=64,
        max_length=512,
        psg_num=1,
        shuffle_id=False,
        max_id=128,
        add_aux_loss=False,
        gpt_style=False,
        use_oracle=False,
        add_label=True,
        add_response=False,
        add_hyperlink=True,
        test=True,
        dialogue_first=True
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Đặt mô hình ở chế độ đánh giá
    model.model.eval()

    # Chuẩn bị các biến theo dõi
    output_id_collect = []
    output_text_collect = []
    true_text_collect = []
    true_id_collect = []
    id_maps = []  # Lưu ánh xạ ID cho mỗi mẫu
    sentence_ids = []  # Lưu ánh xạ sentence_to_id cho mỗi mẫu
    debug_info = []  # Lưu thông tin debug

    # Tiến hành đánh giá
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader))):
            # Chuyển batch lên thiết bị tính toán
            batch = {k: v.cuda() for k, v in batch.items()}

            # Sinh định danh tri thức
            outputs = model.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=16,  # Tăng max_length để đảm bảo ID được sinh đầy đủ
                num_beams=4
            )

            # Lưu ánh xạ ID và sentence_to_id cho mỗi mẫu trong batch hiện tại
            current_id_map = eval_dataset.get_id_map()
            current_sent_map = eval_dataset.get_sentence_to_id_map()
            current_debug = eval_dataset.get_debug_info()

            for i in range(outputs.size(0)):
                if batch_idx * batch_size + i < len(eval_data):
                    id_maps.append(current_id_map)
                    sentence_ids.append(current_sent_map)
                    debug_info.append(current_debug)

            # Giải mã đầu ra và trích xuất ID
            for i in range(outputs.size(0)):
                output_text = model.tokenizer.decode(outputs[i], skip_special_tokens=False)
                # Trích xuất ID từ chuỗi đầu ra
                output_id = model._extract_knowledge_id(output_text)
                output_id_collect.append(output_id)

            # Lấy định danh thực sự từ nhãn
            for i in range(batch['labels'].size(0)):
                label = batch['labels'][i].clone()
                # Thay thế padding token với token thực
                label[label == -100] = model.tokenizer.pad_token_id
                true_text = model.tokenizer.decode(label, skip_special_tokens=False)
                # Trích xuất ID từ nhãn
                true_id = model._extract_knowledge_id(true_text)
                true_id_collect.append(true_id)

    # Đánh giá việc chọn tri thức với cải tiến
    knowledge_matches = []

    for idx, (pred_id, true_id) in enumerate(zip(output_id_collect, true_id_collect)):
        if idx < len(id_maps):
            # So sánh ID với độ linh hoạt cao hơn
            is_match = model._is_id_match(pred_id, true_id)
            knowledge_matches.append(is_match)

            # Lưu thông tin cho debug
            if idx < len(debug_info):
                debug_info[idx].update({
                    'pred_id': pred_id,
                    'true_id': true_id,
                    'is_match': is_match
                })

    # Tính knowledge accuracy
    knowledge_accuracy = (sum(knowledge_matches) / len(knowledge_matches) * 100) if knowledge_matches else 0.0

    # Đánh giá sinh phản hồi
    for idx, example in enumerate(eval_data):
        # Chuẩn bị đầu vào cho việc sinh phản hồi
        response_input = example.copy()

        # Chọn tri thức cho mẫu này
        genks_input = model.prepare_genks_input(
            query=example['context'][-1]['text'] if example.get('context', []) else "",
            docs=[(0, example.get('checked_sentence', ''), example.get('title', ''), 1.0)],
            dialogue_history=[turn['text'] for turn in example.get('context', [])[:-1]] if example.get('context',
                                                                                                       []) else []
        )

        # Sinh phản hồi dựa trên tri thức đã chọn
        response = model.generate_response(
            genks_input=genks_input,
            selected_title=example.get('title', ''),
            selected_sentence=example.get('checked_sentence', ''),
            device='cuda'
        )

        output_text_collect.append(response)
        true_text_collect.append(example.get('labels', [''])[0])

    # Tính toán các metrics dựa trên phản hồi
    # Chuẩn bị dữ liệu cho đánh giá
    refs = [[ref.lower().split()] for ref in true_text_collect]
    hyps = [hyp.lower().split() for hyp in output_text_collect]

    # Tính BLEU
    smoothie = SmoothingFunction().method1
    bleu1 = sum([sentence_bleu([ref[0]], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
                 for ref, hyp in zip(refs, hyps)]) / len(refs)
    bleu4 = sum([sentence_bleu([ref[0]], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
                 for ref, hyp in zip(refs, hyps)]) / len(refs)

    # Tính ROUGE
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores([' '.join(hyp) for hyp in hyps],
                                        [' '.join(ref[0]) for ref in refs], avg=True)
    except:
        # Fallback nếu có vấn đề với ROUGE
        rouge_scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

    # Tính Knowledge F1 (KF1)
    def f1_score(prediction, ground_truths):
        """Tính F1 giữa dự đoán và ground truth"""
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truths[0].split()
        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = len(common)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    kf1 = sum([f1_score(' '.join(hyp), [' '.join(ref[0])]) for hyp, ref in zip(hyps, refs)]) / len(refs)

    # Tổng hợp kết quả
    results = {
        'knowledge_accuracy': knowledge_accuracy,
        'bleu1': bleu1 * 100,
        'bleu4': bleu4 * 100,
        'rouge1': rouge_scores['rouge-1']['f'] * 100,
        'rouge2': rouge_scores['rouge-2']['f'] * 100,
        'rougeL': rouge_scores['rouge-l']['f'] * 100,
        'kf1': kf1 * 100
    }

    # Lưu kết quả đánh giá nếu cần
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Lưu các phản hồi được sinh ra
        with open(output_file.replace('.json', '_responses.txt'), 'w') as f:
            for hyp, ref in zip(output_text_collect, true_text_collect):
                f.write(f"Prediction: {hyp}\n")
                f.write(f"Reference: {ref}\n")
                f.write("-" * 80 + "\n")

        # Lưu thông tin debug
        with open(output_file.replace('.json', '_debug.json'), 'w') as f:
            json.dump(debug_info, f, indent=2)

    return results


def train_improved_rag_genks(model, train_data, eval_data=None, output_dir='ckpt/improved_rag_genks',
                             epochs=5, batch_size=8, accumulation_steps=4, learning_rate=2e-5):
    """
    Huấn luyện mô hình RAG đa giai đoạn kết hợp GENKS cải tiến

    Args:
        model: Mô hình ImprovedMultiStageRAGWithGENKS
        train_data: Dữ liệu huấn luyện
        eval_data: Dữ liệu đánh giá
        output_dir: Thư mục lưu mô hình
        epochs: Số epochs
        batch_size: Kích thước batch
        accumulation_steps: Số bước tích lũy gradient
        learning_rate: Tốc độ học

    Returns:
        Mô hình đã huấn luyện
    """
    # Khởi tạo accelerator
    accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps)
    logger.info(f"Đang huấn luyện mô hình với {len(train_data)} mẫu trong {epochs} epochs")

    # Chuẩn bị dữ liệu
    train_dataset = ImprovedGENKSData(
        train_data,
        model.tokenizer,
        context_len=128,
        sent_len=64,
        max_length=512,
        psg_num=1,
        shuffle_id=False,
        max_id=128,
        add_aux_loss=False,
        gpt_style=False,
        use_oracle=True,
        add_label=True,
        add_response=True,
        add_hyperlink=True,
        add_label_to_prefix=False,
        dialogue_first=True,
        knowledge_response=0.0,
        second_id=False,
        drop_null=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Chuẩn bị optimizer và scheduler
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)
    model.model, optimizer, train_dataloader = accelerator.prepare(model.model, optimizer, train_dataloader)
    scheduler = get_constant_schedule(optimizer)
    scheduler = accelerator.prepare(scheduler)

    # Vòng lặp huấn luyện
    for epoch in range(epochs):
        accelerator.wait_for_everyone()
        logger.info(f'Epoch {epoch + 1}/{epochs}')

        # Đặt mô hình ở chế độ huấn luyện
        model.model.train()

        tk0 = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        acc = []

        for batch_idx, batch in enumerate(tk0):
            with accelerator.accumulate(model.model):
                output = model.model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Tính toán độ chính xác
                acc.append((output.logits.argmax(-1) == batch['labels'])[:, 1].float().mean().item())
                losses.append(loss.item())

                # Cập nhật thanh tiến trình
                tk0.set_postfix(loss=sum(losses) / len(losses), acc=sum(acc) / len(acc))
                scheduler.step()

        # Lưu mô hình sau mỗi epoch
        os.makedirs(output_dir, exist_ok=True)
        if accelerator.is_local_main_process:
            accelerator.save(accelerator.unwrap_model(model.model).state_dict(), f'{output_dir}/epoch_{epoch}.pt')

        # Đánh giá nếu có dữ liệu đánh giá
        if eval_data:
            results = improved_evaluate_multi_stage_rag_genks(
                model=model,
                eval_data=eval_data,
                output_file=f'{output_dir}/eval_results_epoch_{epoch}.json'
            )
            logger.info(f"Kết quả đánh giá epoch {epoch}: {results}")

    # Lưu mô hình cuối cùng
    if accelerator.is_local_main_process:
        model.save(output_dir)

    return model


def main():
    """
    Hàm chính để chạy quá trình huấn luyện và đánh giá
    """
    # Thiết lập logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Khởi tạo mô hình
    improved_rag = ImprovedMultiStageRAGWithGENKS(
        model_name='facebook/bart-base',
        retriever_model_name='facebook/dpr-ctx_encoder-single-nq-base',
        query_encoder_name='facebook/dpr-question_encoder-single-nq-base',
        ranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
        top_k_retrieval=100,
        top_k_rerank=10,
        retrieval_method='bm25'
    )

    # Tải dữ liệu
    train_data = json.load(open('/kaggle/input/wizard/train30.json'))
    valid_data = json.load(open('/kaggle/input/wizard/valid_seen.json'))
    test_seen_data = json.load(open('/kaggle/input/wizard/test_seen.json'))
    test_unseen_data = json.load(open('/kaggle/input/wizard/test_unseen.json'))

    # Xây dựng corpus
    corpus = []
    for i, example in enumerate(train_data):
        for title, sentences in example['knowledge'].items():
            for j, sentence in enumerate(sentences):
                doc_id = f"doc_{i}_{title}_{j}"
                corpus.append((doc_id, sentence, title))

    # Xây dựng index cho corpus
    improved_rag.build_corpus_index(corpus)

    # Huấn luyện mô hình
    train_improved_rag_genks(
        model=improved_rag,
        train_data=train_data,
        eval_data=valid_data,
        epochs=5,
        batch_size=4,
        accumulation_steps=8,
        output_dir='/kaggle/working/ckpt/improved_rag_genks'
    )

    # Đánh giá mô hình
    results_seen = improved_evaluate_multi_stage_rag_genks(
        model=improved_rag,
        eval_data=test_seen_data,
        output_file='/kaggle/working/ckpt/improved_rag_genks/results_seen.json'
    )

    results_unseen = improved_evaluate_multi_stage_rag_genks(
        model=improved_rag,
        eval_data=test_unseen_data,
        output_file='/kaggle/working/ckpt/improved_rag_genks/results_unseen.json'
    )

    # Báo cáo kết quả
    logger.info("Kết quả trên WoW Seen:")
    for metric, value in results_seen.items():
        logger.info(f"{metric}: {value:.2f}")

    logger.info("\nKết quả trên WoW Unseen:")
    for metric, value in results_unseen.items():
        logger.info(f"{metric}: {value:.2f}")


if __name__ == '__main__':
    main()