from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import chardet
from sentence_transformers import SentenceTransformer
from .vector_store import FAISSStore
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1' #windows符号链接限制会有warning,这里把warning忽略让程序正常运行


class KnowledgeBase:
    def __init__(self, index_path: str = None):
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.vector_store = FAISSStore(
            dimension=384,  # MiniLM 的维度
            index_path=index_path
        )

    def detect_file_encoding(self, file_path):
        """检测文件编码,有的文件编码不是utf-8"""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

    def load_data(self, csv_path: str):
        """加载带有部门和标题信息的医疗QA数据"""
        # 使用正确的编码读取
        encoding = self.detect_file_encoding(csv_path)
        df = pd.read_csv(csv_path, encoding=encoding)

        processed_chunks = []
        #当前sample的dataset格式为 column1:科室，column2:主题，column3:问，column4:答
        for _, row in df.iterrows():
            # 组合完整文本，包含所有信息
            full_text = (f"科室：{row['department']} "
                         f"主题：{row['title']} "
                         f"问：{row['ask']} "
                         f"答：{row['answer']}")

            # 使用LangChain分割器处理
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["科室：", "主题：", "问：", "答：", "\n", "。", "！", "？"],
                keep_separator=True
            )

            chunks = text_splitter.split_text(full_text)

            for chunk in chunks:
                processed_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'department': row['department'],
                        'title': row['title'],
                        'original_question': row['ask']
                    }
                })

        # 生成embeddings并存储
        texts = [chunk['text'] for chunk in processed_chunks]
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=True)
        self.vector_store.add_texts(processed_chunks, embeddings)

    def search(self, query: str, k: int = 3):
        """
        搜索相关文档
        Args:
            query: 查询文本
            k: 返回的文档数量
        Returns:
            相关文档列表
        """
        # 获取查询文本的向量表示
        query_embedding = self.embedder.encode(query)

        # 使用向量存储进行搜索
        results = self.vector_store.search(query_embedding, k)

        return results

    def save_index(self, path: str):
        """保存向量索引"""
        self.vector_store.save(path)

    def load_index(self, path: str):
        """加载向量索引"""
        self.vector_store.load(path)

# 以下code只是做了一个vector store的测试
if __name__ == '__main__':
    # 初始化知识库
    kb = KnowledgeBase()

    # 加载数据
    kb.load_data('../data/knowledge_base/sample_IM_5000-6000_utf8.csv')

    # 测试检索
    result = kb.search("高血压的症状有哪些？")
    for doc in result:
        print(f"科室: {doc['metadata']['department']}")
        print(f"主题: {doc['metadata']['title']}")
        print(f"内容: {doc['text']}\n")