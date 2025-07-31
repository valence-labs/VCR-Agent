import anndata
from anndata import AnnData
from pydeseq2.dds import DeseqDataSet

def regulates_expression():
    NotImplemented


def TxSimilarity():
    return

def get_expression_data(h5ad_path) -> AnnData:
    adata = anndata.read_h5ad(h5ad_path)

def comput_DE(control, )