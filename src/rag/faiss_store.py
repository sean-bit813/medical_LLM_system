import os
import pickle

import faiss
import numpy as np
from typing import List, Dict, Any


class FAISSStore:
    def __init__(self, dimension: int, index_path: str = None):
        if index_path and os.path.exists(index_path):
            # 如果提供了索引路径且文件存在，则加载现有索引
            self.index = faiss.read_index(index_path)
        else:
            # 否则创建新索引
            self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.index_path = index_path

    def add_texts(self, processed_chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        添加文本和对应的向量到存储

        Args:
            processed_chunks: 包含文本和元数据的字典列表
            embeddings: 文本对应的向量表示，numpy数组
        """
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        # 确保向量是float32类型
        vectors = embeddings.astype('float32')
        self.index.add(vectors)
        self.chunks.extend(processed_chunks)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """
        搜索最相似的文档

        Args:
            query_embedding: 查询文本的向量表示
            k: 返回的结果数量
        """
        # 确保查询向量格式正确
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')

        # 执行搜索
        D, I = self.index.search(query_embedding, k)

        results = []
        for i, dist in zip(I[0], D[0]):
            if i != -1:  # FAISS可能返回-1表示未找到足够多的结果
                chunk = self.chunks[i]
                results.append({
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'score': float(dist)
                })

        return results

    def save(self, path: str = None):
        try:
            save_path = path or self.index_path
            if save_path:
                # 保存FAISS索引
                faiss.write_index(self.index, save_path)
                # 保存文本数据
                with open(save_path + '.chunks', 'wb') as f:
                    pickle.dump(self.chunks, f)
                print(f"Successfully saved index to {save_path}")
        except Exception as e:
            print(f"Error saving index: {e}")

    def load(self, path: str):
        try:
            self.index = faiss.read_index(path)
            with open(path + '.chunks', 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Successfully loaded index from {path}")
        except Exception as e:
            print(f"Error loading index: {e}")