import os
import requests
from Bio import PDB
from requests.adapters import HTTPAdapter
from urllib3 import Retry


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class ModifiedSingleChainSelect(PDB.Select):#函数未被使用
    def __init__(self, chain_letter):
        self.chain_letter = chain_letter

    def accept_atom(self, atom):
        return atom.get_parent().get_id()[0] == ' '

    def accept_chain(self, chain):
        return chain.id == self.chain_letter


def download_file(url, local_filename, retries=3, backoff_factor=0.3):#要下载文件的url，保存文件的路径，设置重试的次数，每次重试之间的等待时间倍增系数
    """使用requests库下载文件，并增加了重试逻辑。"""
    # 设置重试策略
    session = requests.Session()#复用TCP提高效率
    retries = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[500, 502, 503, 504])#重试三次，第一次重试等待0.3秒，第二次等待0.6秒，服务器返回500等数字时进行重试
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        with session.get(url, stream=True, timeout=10) as r:#打开文件并写入
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")


def download_and_process_pdb(pdb_id, pdb_storage_dir, io):
    ensure_dir(pdb_storage_dir)
    pdb_file_path = os.path.join(pdb_storage_dir, f"{pdb_id.lower()}.pdb")

    if not os.path.exists(pdb_file_path):
        print(f"Downloading PDB structure '{pdb_id}'...")
        custom_url = f"https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdb_id.lower()}/?scheme=chothia"#抗体pdb来源sabdab数据库
        download_file(custom_url, pdb_file_path)

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_file_path)
    except Exception as e:
        print(f"Error processing PDB file {pdb_file_path}: {e}")
        return


def download_antibody_cdr(ppi, config):
    pdb_storage_dir = config['dirs']['CDR']
    ensure_dir(pdb_storage_dir)#路径是否存在，不存在就创建

    io = PDB.PDBIO()

    antigen, antibody = ppi.split(',')
    antigen_id, antigen_chain = antigen.split('_')
    antibody_id, antibody_chain = antibody.split('_')
    download_and_process_pdb(antibody_id, pdb_storage_dir, io)
