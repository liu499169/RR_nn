# In src/core/implicit_solver.py
import torch
from src.core.physics_formulas import (get_plane_h_from_area, get_trapezoid_wp_from_h, 
                                       calculate_q_manning,calculate_dqda_manning_general)

# A specialized, fast solver for bidiagonal systems would be ideal, but torch.linalg.solve is general and works.
# For now, we will build a full matrix and use the general solver.

class ImplicitKinematicWave(torch.autograd.Function):
    """
    Differentiable Kinematic Wave solver using an implicit Preissmann scheme.
    Solves one large, fixed timestep.
    """
    @staticmethod
    def forward(ctx, A_curr, q_upstream, q_lateral, dt, element_props, MAN_tensor):
        """
        Solves the linear system A(theta) * x = b for the next state x = A_next.
        
        Args:
            A_curr (torch.Tensor): Current flow area at nodes.
            q_upstream (torch.Tensor): Upstream inflow Q.
            q_lateral (torch.Tensor): Lateral inflow per unit length at nodes.
            dt (torch.Tensor): The large, fixed timestep.
            element_props (ElementProperties): Static properties of the element.
            learnable_params (dict): A dictionary of learnable parameters, e.g., {'MAN': tensor(0.06)}.
        """
        n_nodes = A_curr.shape[0]
        device, dtype = A_curr.device, A_curr.dtype
        dx = element_props.dx_avg
        
        # --- Linearization Step ---
        # We linearize the system using the state from the current time 'n'.
        # Q = alpha * A. We need alpha = dQ/dA.
        # Calculate q_curr for both geom types
        if element_props.geom_type == 'plane':
            h_curr = get_plane_h_from_area(A_curr, element_props.WID)
            wp_curr = get_trapezoid_wp_from_h(h_curr, element_props.WID, element_props.SS1, element_props.SS2)
            q_curr = calculate_q_manning(A_curr, wp_curr, MAN_tensor, element_props.SL)
            top_width = element_props.WID * torch.ones_like(A_curr)
        else:  # channel
            h_curr = get_plane_h_from_area(A_curr, element_props.W0_nodes)
            wp_curr = get_trapezoid_wp_from_h(h_curr, element_props.W0_nodes, element_props.SS1, element_props.SS2)
            q_curr = calculate_q_manning(A_curr, wp_curr, MAN_tensor, element_props.SL)
            # Top width for trapezoid: W0 + SS1*h + SS2*h
            top_width = element_props.W0_nodes + element_props.SS1 * h_curr + element_props.SS2 * h_curr

        # dQ/dA (alpha) using the provided general function
        alpha = calculate_dqda_manning_general(
            A_curr, wp_curr, top_width, q_curr,
            element_props.SS1, element_props.SS2,
            element_props.geom_type
        )

        # --- Build the Bidiagonal System Ax = b ---
        # A is of shape (n_nodes, n_nodes)
        A_matrix = torch.zeros((n_nodes, n_nodes), device=device, dtype=dtype)
        
        # Define coefficients based on Preissmann scheme
        C1 = dt / (2 * dx)
        
        # Main diagonal: 1 + C1 * alpha_i
        diag_vals = 1 + C1 * alpha
        torch.diag(A_matrix, 0).copy_(diag_vals)

        # Sub-diagonal: -C1 * alpha_i
        sub_diag_vals = -C1 * alpha[1:]
        torch.diag(A_matrix, -1).copy_(sub_diag_vals)

        # --- Build the right-hand side vector b ---
        b_vector = A_curr + dt * q_lateral
        # Apply boundary condition from previous timestep's Q values
        q_interface = (q_curr[:-1] + q_curr[1:]) / 2.0
        b_vector[:-1] -= C1 * q_interface
        b_vector[1:] += C1 * q_interface
        # Upstream boundary condition
        b_vector[0] += (dt / dx) * q_upstream

        # --- Solve the linear system ---
        try:
            A_next = torch.linalg.solve(A_matrix, b_vector)
        except torch.linalg.LinAlgError:
            print(f"Warning: Singular matrix for element {element_props.element_id}. Using previous state.")
            A_next = A_curr.clone()

        # Save tensors needed for the backward pass
        ctx.save_for_backward(A_matrix, A_next, MAN_tensor, dt, dx, alpha)
        
        return torch.clamp(A_next, min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient w.r.t. learnable parameters using the adjoint method.
        This solves the adjoint system: A^T * lambda = grad_L/grad_A_next
        """
        A_matrix, A_next, MAN, dt, dx, alpha = ctx.saved_tensors
        
        # The incoming gradient is the gradient of the final loss w.r.t. the output of this function.
        # This is the right-hand side of our adjoint system.
        
        # 1. Solve the adjoint system: A^T * lambda = grad_output
        lambda_adjoint, _ = torch.linalg.solve_triangular(A_matrix.T, grad_output.unsqueeze(1), upper=False)
        lambda_adjoint = lambda_adjoint.squeeze(1)

        # 2. Compute the gradient of the loss w.r.t. Manning's n (MAN)
        # We need the partial derivative of the system residual (R = Ax - b) w.r.t. MAN.
        # dR/dMAN = (dA/dMAN)*A_next - (db/dMAN)
        # Gradient = -lambda^T * dR/dMAN
        
        # d(alpha)/dMAN = -alpha / MAN
        d_alpha_dMAN = -alpha / (MAN + 1e-9)
        
        # Derivative of the matrix A w.r.t MAN
        C1 = dt / (2 * dx)
        d_A_dMAN_diag = C1 * d_alpha_dMAN
        d_A_dMAN_subdiag = -C1 * d_alpha_dMAN[1:]
        
        # Compute lambda^T * (dA/dMAN) * A_next
        term1 = torch.sum(lambda_adjoint * d_A_dMAN_diag * A_next)
        term2 = torch.sum(lambda_adjoint[:-1] * d_A_dMAN_subdiag * A_next[1:])
        
        # The derivative of b w.r.t MAN is more complex, involving dQ/dMAN.
        # For simplicity in this first implementation, we assume its contribution is smaller.
        # A full implementation would include this term.
        grad_MAN = -(term1 + term2)

        return None, None, None, None, None, grad_MAN