from src.loss_fns.services.dag_loss import brute_force_dag_loss
from src.loss_fns.services.dag_loss_efficient import dag_loss_raw, process_dp
import torch


def test_brute_force_dag_loss_multipath():
    transition_matrix = torch.tensor(
        [
            [0, 0.7, 0.3, 0, 0, 0],
            [0, 0, 0.6, 0.3, 0.1, 0],
            [0, 0, 0, 0.2, 0, 0.8],
            [0, 0, 0, 0, 0.5, 0.5],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    emission_probs = torch.tensor(
        [
            [0.9, 0, 0, 0.1, 0, 0],
            [0.3, 0.6, 0, 0.1, 0, 0],
            [0, 0.2, 0.8, 0, 0, 0],
            [0.1, 0.1, 0, 0.6, 0, 0.2],
            [0, 0, 0.1, 0, 0.9, 0],
            [0, 0, 0, 0.2, 0, 0.8],
        ]
    )
    emp_log = torch.log(emission_probs)
    t_log = torch.log(transition_matrix)
    target_seq = torch.tensor([0, 1, 2, 5])
    loss, paths = brute_force_dag_loss(
        transition_matrix=t_log,
        emissions=emp_log,
        target_sequence=target_seq,
    )

    expected_prob = 0.1191456
    result = torch.exp(-loss)
    expected_prob = torch.tensor(expected_prob)

    assert torch.allclose(result, expected_prob, atol=1e-5)


def test_brute_force_dag_loss_single_path():
    transition_matrix = torch.tensor(
        [
            [0, 0.3, 0.7, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    emission_probs = torch.tensor(
        [
            [0.3823, 0.3522, 0.2655],
            [0.2795, 0.4328, 0.2877],
            [0.3601, 0.4045, 0.2354],
            [0.3799, 0.2209, 0.3992],
            [0.3523, 0.3474, 0.3002],
            [0.2277, 0.4765, 0.2958],
            [0.2893, 0.4171, 0.2936],
            [0.4688, 0.2246, 0.3066],
            [0.4615, 0.2844, 0.2541],
        ]
    )
    target_seq = torch.tensor([0, 1, 0, 2, 2, 0])
    t_log = torch.log(transition_matrix)
    emp_log = torch.log(emission_probs)
    loss, paths = brute_force_dag_loss(
        transition_matrix=t_log,
        emissions=emp_log,
        target_sequence=target_seq,
    )

    expected_prob = 8.42637e-4
    result = torch.exp(-loss)
    expected_prob = torch.tensor(expected_prob)

    assert torch.allclose(result, expected_prob, atol=1e-5)


def test_brute_force_dag_loss_multipath_square_transition_matrix():
    transition_matrix = torch.tensor(
        [
            [0, 0.7, 0.3, 0, 0, 0],
            [0, 0, 0.6, 0.3, 0.1, 0],
            [0, 0, 0, 0.2, 0, 0.8],
            [0, 0, 0, 0, 0.5, 0.5],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    emission_probs = torch.tensor(
        [
            [0.9, 0, 0, 0.1, 0, 0],
            [0.3, 0.6, 0, 0.1, 0, 0],
            [0, 0.2, 0.8, 0, 0, 0],
            [0.1, 0.1, 0, 0.6, 0, 0.2],
            [0, 0, 0.1, 0, 0.9, 0],
            [0, 0, 0, 0.2, 0, 0.8],
        ]
    )
    emp_log = torch.log(emission_probs)
    t_log = torch.log(transition_matrix)
    target_seq = torch.tensor([0, 1, 2, 5])
    loss, paths = brute_force_dag_loss(
        transition_matrix=t_log,
        emissions=emp_log,
        target_sequence=target_seq,
    )

    expected_prob = 0.1191456
    result = torch.exp(-loss)
    expected_prob = torch.tensor(expected_prob)

    assert torch.allclose(result, expected_prob, atol=1e-5)


def test_brute_force_dag_loss_single_path_square_transition_matrix():
    transition_matrix = torch.tensor(
        [
            [0, 0.3, 0.7, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    emission_probs = torch.tensor(
        [
            [0.3823, 0.3522, 0.2655],
            [0.2795, 0.4328, 0.2877],
            [0.3601, 0.4045, 0.2354],
            [0.3799, 0.2209, 0.3992],
            [0.3523, 0.3474, 0.3002],
            [0.2277, 0.4765, 0.2958],
            [0.2893, 0.4171, 0.2936],
            [0.4688, 0.2246, 0.3066],
            [0.4615, 0.2844, 0.2541],
        ]
    )
    target_seq = torch.tensor([0, 1, 0, 2, 2, 0])
    t_log = torch.log(transition_matrix)
    emp_log = torch.log(emission_probs)
    loss, paths = brute_force_dag_loss(
        transition_matrix=t_log,
        emissions=emp_log,
        target_sequence=target_seq,
    )

    expected_prob = 8.42637e-4
    result = torch.exp(-loss)
    expected_prob = torch.tensor(expected_prob)

    assert torch.allclose(result, expected_prob, atol=1e-5)


# def test_dag_loss_multipath():
#     transition_matrix = torch.tensor(
#         [
#             [0, 0.7, 0.3, 0, 0, 0],
#             [0, 0, 0.6, 0.3, 0.1, 0],
#             [0, 0, 0, 0.2, 0, 0.8],
#             [0, 0, 0, 0, 0.5, 0.5],
#             [0, 0, 0, 0, 0, 1],
#             [0, 0, 0, 0, 0, 0],
#         ]
#     )
#     emission_probs = torch.tensor(
#         [
#             [0.9, 0, 0, 0.1, 0, 0],
#             [0.3, 0.6, 0, 0.1, 0, 0],
#             [0, 0.2, 0.8, 0, 0, 0],
#             [0.1, 0.1, 0, 0.6, 0, 0.2],
#             [0, 0, 0.1, 0, 0.9, 0],
#             [0, 0, 0, 0.2, 0, 0.8],
#         ]
#     )
#     emp_log = torch.log(emission_probs)
#     t_log = torch.log(transition_matrix)
#     target_seq = torch.tensor([0, 1, 2, 5])

#     emp_log_batch = emp_log.unsqueeze(0)
#     t_log_batch = t_log.unsqueeze(0)
#     target_seq_batch = target_seq.unsqueeze(0)

#     dp = dag_loss_raw(
#         targets=target_seq_batch,
#         transition_matrix=t_log_batch,
#         emission_probs=emp_log_batch,
#     )

#     target_len = torch.tensor([4])
#     vertex_len = torch.tensor([6])

#     result = process_dp(dp, target_len, vertex_len)
#     expected_prob = 0.1191456
