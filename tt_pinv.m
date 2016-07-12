function [M,S,N] = tt_pinv(tt,l)
%% This script is written according to "Tensor-based dynamic mode
%% decomposition" algorithm authored by Stenfan Klus.
%% Input:
%% tt: input train tensor
%% core_number: ??? not sure about what it means yet
%% Output:
%% Compute pseudo-inverse of reshaped tt at core number nl

    d=tt.d; % dimensionality of tensor
    r=tt.r; % rank of tensor
    cr=tt.core; % tensor core
    ps=tt.ps; % core position in tt.core
    n=tt.n;

    % Step 2
    [tt,rm_lo] = qr_l(tt,'LR',l-1); % lo: left-orthonormization
    [tt,rm_ro] = qr_l(tt,'RL',l+1); % ro: right-orthonormization

    % Step 3
    % get x^l (core l)
    core_l = cr(ps(l):ps(l+1)-1);
    % reshape x^l in with [r(l)*n(l),r(l+1)]
    core_l = reshape(core_l,[r(l)*n(l),r(l+1)]);    
    % SVD of x^l
    [U,S,V] = svd(core_l,'econ');

    % Step 4
    s = size(S,1); % new rank at l
    cr = [cr(1:ps(l)-1);U(:);cr(ps(l+1):end)]; % populate new core
    ps(l+1:end) = ps(l+1:end) - (r(l+1)-s)*r(l)*n(l);
    r(l+1) = s; % assign new rank     
        
    % Step 5
    % get x^(l+1) (core (l+1))
    core_lp1 = cr(ps(l+1):ps(l+2)-1);
    % reshape x^(l+1)
    core_lp1 = reshape(core_lp1,[r(l+1),n(l+1)*r(l+2)]);
    % compute z
    z = V'*core_lp1;
    % populate core (l+1)
    cr = [cr(1:ps(l+1)-1);z(:);cr(ps(l+2):end)];
    % update position
    ps(l+2:end) = ps(l+2:end) - (r(l+1)-s)*n(l+1)*r(l+2);

    % Step 6: ( Expecting bugs here )
    % Done in Step 5 and 4
    
    % Step 7
    M = tt_tensor;
    M.d = l; M.r = r(1:l+1); M.n = n(1:l);  M.core = cr(1:ps(l+1)); M.ps = ps(1:l+1);
    N = tt_tensor;
    N.d = d-l; N.r = r(l+1:end); N.n = n(l+1:end);
    N.core = cr(ps(l+1):end); N.ps = ps(l+1:end) - ps(l+1) + 1;
    
% $$$     M = tt_matrix(M,M.n,r(l));
% $$$     N = tt_matrix(N,r(l),N.n);
end
