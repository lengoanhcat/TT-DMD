function [Phi,Lambda] = tt_dmd(ttX,ttY,defDMD)
%% This script is written according to "Tensor-based dynamic mode
%% decomposition" algorithm authored by Stenfan Klus.
%% Input:
%% ttX: input sequence X in TT-format
%% ttY: input sequence Y in TT-format
%% defDMD: definition of DMD Pro(jection)/Exa(ct)
%% core_number: ??? not sure about what it means yet
%% Output:
%% Compute pseudo-inverse of reshaped tt at core number nl
    
    % Compute pseudo-inverse of A
    [M,S,N] = tt_pinv(ttX,ttX.d-1);

    % Get range (1:d) of ttX and ttY
    ttX_d = tt_tensor;
    ttX_d.core = ttX.core(1:ttX.ps(end-1));
    ttX_d.ps = ttX.ps(1:end-1);
    ttX_d.d = ttX.d-1;
    ttX_d.n = ttX.n(1:end-1);
    ttX_d.r = ttX.r(1:end-1);
    
    ttY_d = tt_tensor;
    ttY_d.core = ttY.core(1:ttY.ps(end-1));
    ttY_d.ps = ttY.ps(1:end-1);
    ttY_d.d = ttY.d-1;
    ttY_d.n = ttY.n(1:end-1);
    ttY_d.r = ttY.r(1:end-1);    

    % Get core (d+1) of ttX and ttY
    
    ttX_dp1 = tt_tensor;
    ttX_dp1.core = ttX.core(ttX.ps(end-1):ttX.ps(end)-1);
    ttX_dp1.ps = ttX.ps(end-1:end)-ttX.ps(end-1) + 1;
    ttX_dp1.d = 1;
    ttX_dp1.n = ttX.n(end);
    ttX_dp1.r = ttX.r(end-1:end);
    
    ttY_dp1 = tt_tensor;
    ttY_dp1.core = ttY.core(ttY.ps(end-1):ttY.ps(end)-1);
    ttY_dp1.ps = ttY.ps(end-1:end) - ttY.ps(end-1) + 1;
    ttY_dp1.d = 1;
    ttY_dp1.n = ttY.n(end);
    ttY_dp1.r = ttY.r(end-1:end);
    
    % Compute M^T . P - contraction of P core with orthonormized M^T
    MtP = dot(ctranspose(ttX_d),ttY_d);
    % MtP = dot_transpose(ttX_d,true,ttY_d,false);
    
    % Compute Q . N^T - contraction of  core with orthonormized N^T
    QNt = dot(ttY_dp1,ctranspose(ttX_dp1));
    % QNt = dot_transpose(ttY_dp1,false,ttX_dp1,true);

    invS = diag(1.0/diag(S));
    
    % Compute reduced matrix A - Potential Big Bugs !!! 
    A = MtP*QNt*invS;

    % Compute W - eignvector , lambda - eignvalue of reduced A
    % tol = 1e-10; [W,lambda] = dmrg_eig(tt_A,tol);
    [W, Lambda] = eig(A);
    
    % Compute TT-DMD Modes
    % - Projected DMD Mode
    % !!! Bug Aware
    if isequal('Pro',defDMD)
        tt_Phi = M;
        tt_Phi.core = [tt_Phi.core; W(:)];
        tt_Phi.d = tt_Phi.d + 1;
        tt_Phi.n = [tt_Phi.n; size(W,2)];
        tt_Phi.r = [tt_Phi.r; 1];
        tt_Phi.ps = [tt_Phi.ps; tt_Phi.ps(end)+numel(W)];
    end
    
    % - Exact DMD Mode
    if isequal('Exa',defDMD)
        tt_Phi = ttY;
        tmp = QNt*invS*W*inv(Lambda);
        tt_Phi(tt_Phi.ps(end-1):tt_Phi.ps(end)-1)=tmp(:);
    end
    
    % Compute TT-DMD Lambda
    Phi = full(tt_Phi,tt_Phi.n');
end
