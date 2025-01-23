#转换dataset文件编码为utf-8
def convert_to_utf8(input_file: str, output_file: str):
    """将文件转换为UTF-8编码"""
    encodings = ['utf-8', 'gb18030', 'gbk', 'gb2312']

    for encoding in encodings:
        try:
            # 尝试读取
            with open(input_file, 'r', encoding=encoding) as f:
                content = f.read()

            # 如果成功读取，写入UTF-8文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Successfully converted from {encoding} to UTF-8")
            return True

        except UnicodeDecodeError:
            continue

    return False

convert_to_utf8('sample_IM_5000-6000.csv',
                'sample_IM_5000-6000_utf8.csv')