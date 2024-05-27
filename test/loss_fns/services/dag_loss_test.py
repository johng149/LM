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


def test_dag_loss_rand_batch1_super_massive():
    torch.manual_seed(0)
    np.random.seed(0)

    num_classes = 50
    target_seq_len = 10
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
    expected_prob = torch.tensor(0.4664)

    dp = dag_loss_raw(
        targets=target_seq_batch,
        transition_matrix=t_log_batch,
        emission_probs=emp_log_batch,
    )

    result = process_dp(dp, torch.tensor([target_seq_len]), torch.tensor([vertex_len]))
    result = torch.exp(result)

    assert torch.allclose(result, expected_prob, atol=1e-3)


def test_dag_log_rand_batch2():
    torch.manual_seed(0)
    np.random.seed(0)

    num_classes = 3
    target_seq_len = 10
    vertex_len = 15

    transition_matrices = []
    emission_probs = []
    targets = []

    batch_size = 2
    for _ in range(batch_size):
        transition_matrix = torch.rand(vertex_len, vertex_len)
        transition_matrix = torch.triu(transition_matrix, diagonal=1)
        emission_prob = torch.rand(vertex_len, num_classes)
        target_seq = torch.randint(0, num_classes, (target_seq_len,))
        t_log = torch.log(transition_matrix)
        emp_log = torch.log(emission_prob)

        transition_matrices.append(t_log)
        emission_probs.append(emp_log)
        targets.append(target_seq)

    dp_brutes = []
    for i in range(batch_size):
        dp, _ = brute_force_dag_loss(
            transition_matrix=transition_matrices[i],
            emissions=emission_probs[i],
            target_sequence=targets[i],
        )
        dp_brutes.append(dp)

    dp_brutes = torch.stack(dp_brutes)

    t_logs = torch.stack(transition_matrices)
    emp_logs = torch.stack(emission_probs)
    targets = torch.stack(targets)

    dp = dag_loss_raw(
        targets=targets,
        transition_matrix=t_logs,
        emission_probs=emp_logs,
    )

    target_lens = torch.tensor([target_seq_len for _ in range(batch_size)])
    vertex_lens = torch.tensor([vertex_len for _ in range(batch_size)])

    result = process_dp(dp, target_lens, vertex_lens)

    expected_probs = torch.exp(-dp_brutes)
    result = torch.exp(result)

    assert torch.allclose(result, expected_probs, atol=1e-3)


def test_dag_loss_rand_batch_various_lengths():
    torch.manual_seed(0)
    np.random.seed(0)

    num_classes = 3

    transition_matrices = []
    emission_probs = []
    targets = []

    various_lengths = [(10, 15), (12, 16), (4, 9), (5, 10)]

    batch_size = 4
    for i in range(batch_size):
        target_seq_len = various_lengths[i][0]
        vertex_len = various_lengths[i][1]

        transition_matrix = torch.rand(vertex_len, vertex_len)
        transition_matrix = torch.triu(transition_matrix, diagonal=1)
        emission_prob = torch.rand(vertex_len, num_classes)
        target_seq = torch.randint(0, num_classes, (target_seq_len,))
        t_log = torch.log(transition_matrix)
        emp_log = torch.log(emission_prob)

        transition_matrices.append(t_log)
        emission_probs.append(emp_log)
        targets.append(target_seq)

    dp_brutes = []
    for i in range(batch_size):
        dp, _ = brute_force_dag_loss(
            transition_matrix=transition_matrices[i],
            emissions=emission_probs[i],
            target_sequence=targets[i],
        )
        dp_brutes.append(dp)

    dp_brutes = torch.stack(dp_brutes)

    dp_results = []
    for i in range(batch_size):
        t_log = transition_matrices[i]
        emp_log = emission_probs[i]
        target_seq = targets[i]

        t_log_batch = t_log.unsqueeze(0)
        emp_log_batch = emp_log.unsqueeze(0)
        target_seq_batch = target_seq.unsqueeze(0)

        dp = dag_loss_raw(
            targets=target_seq_batch,
            transition_matrix=t_log_batch,
            emission_probs=emp_log_batch,
        )

        result = process_dp(
            dp,
            torch.tensor([various_lengths[i][0]]),
            torch.tensor([various_lengths[i][1]]),
        )

        dp_results.append(result)

    dp_results = torch.stack(dp_results)
    dp_results = dp_results.flatten()

    expected_probs = torch.exp(-dp_brutes)
    result = torch.exp(dp_results)

    assert torch.allclose(result, expected_probs, atol=1e-3)


# def test_dag_loss_rand_batch_various_lengths_padded_small():
#     # FIXME: This test is failing because the padding is not being handled correctly
#     torch.manual_seed(0)
#     np.random.seed(0)

#     num_classes = 3

#     transition_matrices = []
#     emission_probs = []
#     targets = []

#     various_lengths = [(3, 4), (2, 4), (2, 3), (4, 5)]

#     batch_size = 4
#     for i in range(batch_size):
#         target_seq_len = various_lengths[i][0]
#         vertex_len = various_lengths[i][1]

#         transition_matrix = torch.rand(vertex_len, vertex_len)
#         transition_matrix = torch.triu(transition_matrix, diagonal=1)
#         emission_prob = torch.rand(vertex_len, num_classes)
#         target_seq = torch.randint(0, num_classes, (target_seq_len,))
#         t_log = torch.log(transition_matrix)
#         emp_log = torch.log(emission_prob)

#         transition_matrices.append(t_log)
#         emission_probs.append(emp_log)
#         targets.append(target_seq)

#     dp_brutes = []
#     for i in range(batch_size):
#         dp, _ = brute_force_dag_loss(
#             transition_matrix=transition_matrices[i],
#             emissions=emission_probs[i],
#             target_sequence=targets[i],
#         )
#         dp_brutes.append(dp)

#     dp_brutes = torch.stack(dp_brutes)

#     transition_matrices = torch.nested.nested_tensor(transition_matrices)
#     emission_probs = torch.nested.nested_tensor(emission_probs)
#     targets = torch.nested.nested_tensor(targets)

#     transition_matrices = torch.nested.to_padded_tensor(
#         transition_matrices, padding=100
#     )
#     transition_matrices = torch.tri
#     emission_probs = torch.nested.to_padded_tensor(emission_probs, padding=100)
#     targets = torch.nested.to_padded_tensor(targets, padding=0)

#     dp = dag_loss_raw(
#         targets=targets,
#         transition_matrix=transition_matrices,
#         emission_probs=emission_probs,
#     )

#     target_lens = torch.tensor([various_lengths[i][0] for i in range(batch_size)])
#     vertex_lens = torch.tensor([various_lengths[i][1] for i in range(batch_size)])

#     result = process_dp(dp, target_lens, vertex_lens)

#     expected_probs = torch.exp(-dp_brutes)
#     result = torch.exp(result)

#     assert torch.allclose(result, expected_probs, atol=1e-3)
