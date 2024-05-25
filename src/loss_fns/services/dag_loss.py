import torch
from torch import Tensor


class EmissionMan:
    def __init__(self, emission_probs: Tensor):
        self.ep = emission_probs

    def get_prob(self, node_pos: int, state: int) -> float:
        return self.ep[state][node_pos]


class DFS:
    def __init__(self):
        self.acc = []
        self.t_m = None
        self.rows = 0
        self.cols = 0
        self.e_m = None
        self.t_s = None
        self.t_s_len = 0

    def reset(self, transition_matrix, emission_man, target_seq):
        self.acc = []
        self.t_m = transition_matrix
        self.rows, self.cols = transition_matrix.shape
        self.e_m = emission_man
        self.t_s = target_seq
        self.t_s_len = len(target_seq)

    def dfs(self, seq_pos: int, row: int, total_len: int, prob: float):
        if total_len >= self.t_s_len:
            if total_len == self.t_s_len and row >= self.rows:
                self.acc.append(prob)
            elif total_len > self.t_s_len:
                # this should never happen
                raise ValueError("total_len > target_seq_len")
            return
        if row >= self.rows:
            return
        current_row = self.t_m[row]
        for col, transition_prob in enumerate(current_row):
            new_row = col
            new_prob = (
                prob
                + transition_prob
                + self.e_m.get_prob(node_pos=new_row, state=self.t_s[seq_pos])
            )
            new_seq_pos = seq_pos + 1
            new_total_len = total_len + 1
            if new_row <= row:
                continue
            self.dfs(new_seq_pos, new_row, new_total_len, new_prob)

    def search(self):
        seq_pos = 0
        total_len = 0
        prob = 0
        row = -1
        prob += self.e_m.get_prob(seq_pos, self.t_s[seq_pos])
        row += 1
        seq_pos += 1
        total_len += 1
        self.dfs(seq_pos, row, total_len, prob)
        return self.acc


def brute_force_dag_loss(
    transition_matrix: Tensor, emissions: Tensor, target_sequence: Tensor
):
    """
    Brute force implementation of the DAG loss function, it calculates the
    summed probability of all possible paths through the DAG that generate
    the target sequence. It assumes no batch dimension and no padding.

    @param transition_matrix: Tensor of shape (N-1, N) where N is the number of
    nodes in the DAG. It should contain log probabilities of transitioning
    from one node to another. The reason the first dimension is N-1 is because
    we can assume that once we reach the last node, there are no more valid
    paths to take.
    @param emissions: Tensor of shape (N, C) where N is the number of nodes in
    the DAG and C is the number of classes that can be emitted by each node.
    It should contain log probabilities of emitting each class at each node.
    @param target_sequence: Tensor of shape (T,) where T is the length of the
    target sequence. It should contain values between 0 and C-1, where C is the
    number of classes that can be emitted by each node.

    @return: The negative log probability of the target sequence given the DAG
    """
    em = EmissionMan(emission_probs=emissions.T)
    dfs = DFS()
    dfs.reset(
        transition_matrix=transition_matrix, emission_man=em, target_seq=target_sequence
    )
    acc = dfs.search()
    return -torch.logsumexp(torch.tensor(acc), dim=0)
