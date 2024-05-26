from src.loss_fns.services.dag_loss import brute_force_dag_loss
from src.loss_fns.services.dag_loss_efficient import dag_loss_raw, process_dp
import torch
import numpy as np


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


def test_dag_loss_multipath():
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

    emp_log_batch = emp_log.unsqueeze(0)
    t_log_batch = t_log.unsqueeze(0)
    target_seq_batch = target_seq.unsqueeze(0)

    dp = dag_loss_raw(
        targets=target_seq_batch,
        transition_matrix=t_log_batch,
        emission_probs=emp_log_batch,
    )

    target_len = torch.tensor([4])
    vertex_len = torch.tensor([6])

    result = process_dp(dp, target_len, vertex_len)
    expected_prob = 0.1191456
    result = torch.exp(result)
    expected_prob = torch.tensor(expected_prob)

    assert torch.allclose(result, expected_prob, atol=1e-5)


def test_dag_loss_single_path():
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

    emp_log = torch.log(emission_probs)
    t_log = torch.log(transition_matrix)

    emp_log_batch = emp_log.unsqueeze(0)
    t_log_batch = t_log.unsqueeze(0)
    target_seq_batch = target_seq.unsqueeze(0)

    dp = dag_loss_raw(
        targets=target_seq_batch,
        transition_matrix=t_log_batch,
        emission_probs=emp_log_batch,
    )

    target_len = torch.tensor([6])
    vertex_len = torch.tensor([9])

    result = process_dp(dp, target_len, vertex_len)
    expected_prob = 8.42637e-4
    result = torch.exp(result)
    expected_prob = torch.tensor(expected_prob)

    assert torch.allclose(result, expected_prob, atol=1e-5)


def test_dag_loss_rand_batch1():
    torch.manual_seed(0)
    np.random.seed(0)

    num_classes = 3
    target_seq_len = 10
    vertex_len = 15

    transition_matrix = torch.rand(vertex_len, vertex_len)
    transition_matrix = torch.triu(transition_matrix, diagonal=1)

    emission_probs = torch.rand(vertex_len, num_classes)

    target_seq = torch.randint(0, num_classes, (target_seq_len,))

    emp_log = torch.log(emission_probs)
    t_log = torch.log(transition_matrix)

    emp_log_batch = emp_log.unsqueeze(0)
    t_log_batch = t_log.unsqueeze(0)
    target_seq_batch = target_seq.unsqueeze(0)

    dp_brute, _ = brute_force_dag_loss(
        transition_matrix=t_log, emissions=emp_log, target_sequence=target_seq
    )

    dp = dag_loss_raw(
        targets=target_seq_batch,
        transition_matrix=t_log_batch,
        emission_probs=emp_log_batch,
    )
    result = process_dp(dp, torch.tensor([target_seq_len]), torch.tensor([vertex_len]))

    expected_prob = torch.exp(-dp_brute)
    result = torch.exp(result)

    assert torch.allclose(result, expected_prob, atol=1e-5)


def test_dag_loss_rand_batch1_massive():
    torch.manual_seed(0)
    np.random.seed(0)

    num_classes = 50
    target_seq_len = 9
    vertex_len = target_seq_len * 3

    transition_matrix = torch.rand(vertex_len, vertex_len)
    transition_matrix = torch.triu(transition_matrix, diagonal=1)

    emission_probs = torch.rand(vertex_len, num_classes)

    target_seq = torch.randint(0, num_classes, (target_seq_len,))

    emp_log = torch.log(emission_probs)
    t_log = torch.log(transition_matrix)

    emp_log_batch = emp_log.unsqueeze(0)
    t_log_batch = t_log.unsqueeze(0)
    target_seq_batch = target_seq.unsqueeze(0)

    # dp_brute, _ = brute_force_dag_loss(
    #     transition_matrix=t_log, emissions=emp_log, target_sequence=target_seq
    # )
    # the above takes a while to run for a test case, so we calculate it
    # ahead of time and hard code it here, it is fine since we set the seed
    dp_brute = torch.tensor(-0.9361)

    dp = dag_loss_raw(
        targets=target_seq_batch,
        transition_matrix=t_log_batch,
        emission_probs=emp_log_batch,
    )
    result = process_dp(dp, torch.tensor([target_seq_len]), torch.tensor([vertex_len]))

    # note that this will result in probability above 1, but that is fine
    # because the transition and emission probabilities are not normalized
    expected_prob = torch.exp(-dp_brute)
    result = torch.exp(result)

    assert torch.allclose(result, expected_prob, atol=1e-3)
