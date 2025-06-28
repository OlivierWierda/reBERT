def split_768_to_384(hidden_states):
    # Split 768D into two 384D branches
    mid_point = hidden_states.size(-1) // 2  # 384
    branch_A = hidden_states[:, :, :mid_point]
    branch_B = hidden_states[:, :, mid_point:]
    return branch_A, branch_B


def split_384_to_192(hidden_states_384):
    """
    Split 384D hidden states into two 192D branches

    Args:
        hidden_states_384: tensor of shape (batch_size, seq_len, 384)

    Returns:
        tuple: (sub_branch_1, sub_branch_2) each of shape (batch_size, seq_len, 192)
    """
    mid_point = hidden_states_384.size(-1) // 2  # 192
    sub_branch_1 = hidden_states_384[:, :, :mid_point]  # First 192 dimensions
    sub_branch_2 = hidden_states_384[:, :, mid_point:]  # Second 192 dimensions

    return sub_branch_1, sub_branch_2

