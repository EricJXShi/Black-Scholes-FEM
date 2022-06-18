%% Define Shape Functions
dx = 1;
syms n 
N1n = 1/2*(1-n); N2n = 1/2*(1+n); % Linear shape functions in local coordinates
dndx = 2/dx; % Relationship between dx and dn in the form of dn/dx
dN1dn = diff(N1n); dN2dn = diff(N2n); % Derivative of shape functions wrt n
dN1dx = dN1dn*dndx; dN2dx = dN2dn*dndx; % Derivative of shape functions wrt x

%% Derive Parts of 'Stiffness' Matrix using Method of Weighted Residuals
k11eA = int(N1n^2, [-1 1]); k12eA = int(N1n*N2n, [-1 1]);
k21eA = int(N2n*N1n, [-1 1]); k22eA = int(N2n^2, [-1 1]);
kA = [k11eA, k12eA; k21eA, k22eA]

k11eC = double(int(dN1dx.^2, [-1 1])); k12eC = double(int(dN1dx.*dN2dx, [-1 1]));
k21eC = double(int(dN2dx.*dN1dx, [-1 1])); k22eC = double(int(dN2dx.^2, [-1 1]));
kC = [k11eC, k12eC; k21eC, k22eC]