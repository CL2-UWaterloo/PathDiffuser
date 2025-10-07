import torch

# Example: batch size of num_a agents (num_a x 2 x 2) rotation matrices
R_your = torch.tensor([[0.866, -0.5],    # Rotation matrix of your agent
                       [0.5, 0.866]])    # shape (2, 2)

R_others = torch.tensor([[[0.866, -0.5],  # Rotation matrices of other agents
                          [0.5, 0.866]],   # shape (num_a, 2, 2)
                         [[1.0, 0.0],
                          [0.0, 1.0]]])

# Positions of agents
p_your = torch.tensor([1.0, 2.0])  # shape (2)
p_others = torch.tensor([[3.0, 4.0],  # shape (num_a, 2)
                         [5.0, 6.0]])

# Step 1: Calculate the relative rotation
R_your_T = R_your.T  # Transpose since R_your is orthogonal
delta_rot = torch.matmul(R_others, R_your_T)  # (num_a, 2, 2)

# Step 2: Calculate the relative translation
init_translation = p_others - p_your  # (num_a, 2)

# Recover the original rotations and translations
# Step 3: Recover the original rotation matrices
R_recovered = torch.matmul(delta_rot, R_your)  # Recovering R_others

# Step 4: Recover the original translations
p_recovered = init_translation + p_your  # Recovering p_others

# Output the original and recovered results
print("Original Rotation Matrices (R_others):", R_others)
print("Recovered Rotation Matrices (R_recovered):", R_recovered)

print("Original Translations (p_others):", p_others)
print("Recovered Translations (p_recovered):", p_recovered)

# Step 5: Check if there is any difference
rotation_error = torch.allclose(R_others, R_recovered, atol=1e-4)
translation_error = torch.allclose(p_others, p_recovered, atol=1e-6)

print("\nIs there any rotation error? ", not rotation_error)
print("Is there any translation error? ", not translation_error)



# import torch

# # Example: batch size of num_a agents (num_a x 2 x 2) rotation matrices
# R_your = torch.tensor([[0.866, -0.5],    # Rotation matrix of your agent
#                        [0.5, 0.866]])    # shape (2, 2)

# R_others = torch.tensor([[[0.866, -0.5],  # Rotation matrices of other agents
#                           [0.5, 0.866]],   # shape (num_a, 2, 2)
#                          [[1.0, 0.0],
#                           [0.0, 1.0]]])

# # Positions of agents
# p_your = torch.tensor([1.0, 2.0])  # shape (2)
# p_others = torch.tensor([[3.0, 4.0],  # shape (num_a, 2)
#                          [5.0, 6.0]])

# # Step 1: Calculate the relative rotation
# R_your_T = R_your.T  # Transpose since R_your is orthogonal
# delta_rot = torch.matmul(R_others, R_your_T)  # (num_a, 2, 2)

# # Step 2: Calculate the relative translation
# init_translation = p_others - p_your  # (num_a, 2)

# # Output results
# print("Delta Rotations (2D):", delta_rot)
# print("Delta Translations (2D):", init_translation)


# import torch

# def rot_angle_2d(mat):
#     """
#     Computes the rotation angle (in radians) for a batch of 2D rotation matrices.
#     mat: shape (..., 2, 2)
#     """
#     eps = 1e-4
#     # The trace is the sum of the diagonal elements
#     cos = (mat[..., 0, 0] + mat[..., 1, 1]) / 2
#     # Clamp to avoid numerical issues with acos
#     cos = cos.clamp(-1 + eps, 1 - eps)
#     angle = torch.acos(cos)
#     return angle

# # Example 2D rotation matrices (batch_size, 2, 2)
# rot_pred = torch.tensor([[[0.866, -0.5],  # cos(30), -sin(30)
#                           [0.5, 0.866]],  # sin(30), cos(30)
#                          [[1.0, 0.0],    # cos(0), -sin(0)
#                           [0.0, 1.0]]])   # sin(0), cos(0)

# rot_gt = torch.tensor([[[1.0, 0.0],  # cos(0), -sin(0)
#                         [0.0, 1.0]], # sin(0), cos(0)
#                        [[0.707, -0.707],  # cos(45), -sin(45)
#                         [0.707, 0.707]]]) # sin(45), cos(45)

# # Calculate relative rotation matrices
# relative_rot = torch.matmul(rot_pred, rot_gt.permute(0, 2, 1))  # (batch_size, 2, 2)

# # Calculate rotation loss (geodesic loss)
# rot_loss = rot_angle_2d(relative_rot).mean()  # Mean of the batch

# # Example 2D positions (batch_size, 2)
# pos_pred = torch.tensor([[3.0, 4.0],
#                          [5.0, 6.0]])

# pos_gt = torch.tensor([[1.0, 2.0],
#                        [2.0, 3.0]])

# # Calculate translation loss (MSE)
# trans_loss = torch.nn.functional.mse_loss(pos_pred, pos_gt)

# # Output the losses
# print("Rotation Loss (2D):", rot_loss.item())
# print("Translation Loss (2D):", trans_loss.item())
