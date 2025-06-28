def split_768_to_384(hidden_states):
    # Split 768D into two 384D branches
    mid_point = hidden_states.size(-1) // 2  # 384
    branch_A = hidden_states[:, :, :mid_point]
    branch_B = hidden_states[:, :, mid_point:]
    return branch_A, branch_B