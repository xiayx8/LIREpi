import torch
import numpy as np
import esm
from pathlib import Path
seqnum = 1


torch.manual_seed(1)


def EsmEmbedding(seqname, fasta, model, batch_converter, device='cpu'):
    if len(fasta) > 1022:  # ESM-2模型输入长度限制
        print(f"Warning: Sequence {seqname} is too long ({len(fasta)} residues); it will be truncated.")

    if any(residue not in alphabet.all_toks for residue in fasta):  # 检查非标准氨基酸
        print(f"Warning: Sequence {seqname} contains non-standard amino acids; they will be treated as gaps.")

    data = [(seqname, fasta.upper())]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    try:
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Error: CUDA out of memory when processing {seqname}. Consider reducing batch size or sequence length.")
            return None
        else:
            raise e

    token_representations = results['representations'][33]
    token_representations = token_representations[:, 1:-1]  # 删除特殊的开始和结束token
    return token_representations


def process_fasta_file(input_file_path, model, batch_converter, device='cpu'):
    seqname = ""
    fasta = ""
    with open(input_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # 序列名称
                if seqname and fasta:  # 如果之前读取过序列，现在处理它
                    embeddings = EsmEmbedding(seqname, fasta, model, batch_converter, device)
                    if embeddings is not None:
                        embeddings = embeddings.squeeze()  # 移除长度为1的维度
                        # 保存嵌入到npy文件
                        output_path = Path('/mnt/Data6/23gsy/SEKD-main/esm') / f'{seqname}.npy'
                        output_path.parent.mkdir(exist_ok=True, parents=True)
                        np.save(output_path, embeddings.cpu().numpy())
                        print(f'Saved embeddings for {seqname} to {output_path}')
                    fasta = ""  # 重置fasta字符串，为下一个序列准备
                seqname = line[1:]  # 去掉">"开始的序列名
            else:
                fasta += line  # 累加序列字符串
        if seqname and fasta:  # 确保最后一个序列也被处理
            embeddings = EsmEmbedding(seqname, fasta, model, batch_converter, device)
            if embeddings is not None:
                embeddings = embeddings.squeeze()  # 移除长度为1的维度
                # 保存嵌入到npy文件
                output_path = Path('/mnt/Data6/23gsy/SEKD-main/esm') / f'{seqname}.npy'
                output_path.parent.mkdir(exist_ok=True, parents=True)
                np.save(output_path, embeddings.cpu().numpy())
                print(f'Saved embeddings for {seqname} to {output_path}')



if __name__ == '__main__':
    # 加载ESM模型和batch_converter
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    input_file_path = '/mnt/Data6/23gsy/SEKD-main/newsematest.fasta'
    process_fasta_file(input_file_path, model, batch_converter, device)
