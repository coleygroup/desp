import faiss
import torch
import torch.nn as nn
import numpy as np
import pickle
from scipy import sparse
from desp.inference.models.fwd_model.fwd_model import FwdTemplRel


class BuildingBlockPredictor(FwdTemplRel):
    """
    Class for building block predictor, which predicts the highest probability building block
    given the concatenation of 3 fingerprints:
        1. the fingerprint of the reactant
        2. the fingerprint of the target molecule
        3. the fingerprint of the reaction template
    The predicted fingerprint is then used to search the Faiss index of building block fingerprints,
    returning the top-k building block SMILES, fingerprints, and scores.
    """

    def __init__(
        self,
        model_path,
        bb_tensor_path,
        bb_mol2idx,
        device="cpu",
    ):
        """
        Args:
            model_path (str): path to a trained model
            bb_tensor_path (str): path to the tensor of building blocks
            bb_mol2idx (dict): dict mapping building blocks to index
            device (str): device to run the KNN on
        """
        # Load the bb model
        bb_checkpoint = torch.load(model_path, map_location="cpu")
        pretrain_args = bb_checkpoint["args"]
        pretrain_args.output_dim = 256
        super().__init__(pretrain_args)
        state_dict = bb_checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)
        self.eval()
        self.device = device

        # Load the building block tensor into Faiss index
        with open(bb_tensor_path, "rb") as f:
            bb_fps = sparse.load_npz(f)

        # bb_tensor = bb_fps.toarray().astype(np.float32)
        # Convert in smaller chunks to avoid memory spike
        chunk_size = 10000  # Adjust this based on your memory/matrix dimensions
        total_rows = bb_fps.shape[0]
        bb_tensor = np.zeros((total_rows, bb_fps.shape[1]), dtype=np.float32)

        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            bb_tensor[i:end_idx] = bb_fps[i:end_idx].toarray()
            if i % (chunk_size * 10) == 0:
                print(f"Processed {i}/{total_rows} rows...")
        
        # self.bb_index = faiss.IndexFlatIP(pretrain_args.output_dim)
        quantizer = faiss.IndexFlatIP(pretrain_args.output_dim)
        self.bb_index = faiss.IndexIVFPQ(
            quantizer, pretrain_args.output_dim, 2048, 32, 8
        )  # TODO: Optimize this with https://www.pinecone.io/learn/series/faiss/product-quantization/
        self.bb_index.nprobe = 64
        res = faiss.StandardGpuResources()
        if self.device != "cpu":
            self.bb_index = faiss.index_cpu_to_gpu(res, int(self.device), self.bb_index)
            faiss.normalize_L2(bb_tensor)
            self.bb_index.train(bb_tensor)
            self.bb_index.add(bb_tensor)
        else:
            faiss.normalize_L2(bb_tensor)
            self.bb_index = quantizer
            self.bb_index.add(bb_tensor)

        self.bb_idx2mol = {v: k for k, v in bb_mol2idx.items()}

    def _predict(self, x):
        """
        Make forward pass of the MLP model to predict the building block.
        """
        preds = self.forward(x)
        preds = nn.Sigmoid()(preds)
        return preds.detach().numpy()

    def get_topk_bb(self, x, k=10):
        """
        Get top-k building block SMILES, fingerprints, and scores given the input fingerprint tensor.
        """
        preds = self._predict(x)
        faiss.normalize_L2(preds)
        _, indices = self.bb_index.search(preds, k)
        bb_smiles = [[self.bb_idx2mol[i] for i in results] for results in indices]
        return bb_smiles
