clc, clear, close all
%% Sanity Check with Assignment 3 Problem
%% Given Data
D = 6*10^-12; % Apparent Diffusion Constant (m^2/s)
n_time = 100;
t_total = n_time*24*60*60; % Total time (seconds)
L = 0.1; % Total depth (m)
n_blocks = 100; % Number of grid blocks
n_nodes = n_blocks+1;
dx = L/n_blocks; % Space step
dt = t_total/n_time; % 1 Day time step (seconds)
V = 0; % Advection Constant
phi_1 = 1; % Concentration at the surface - Boundary Condition (%)
F = zeros(n_nodes, 1);
F(1) = 1;

x_coord = zeros(n_nodes, 1);
for i = 1:n_nodes
    x_coord(i) = (i-1)*dx;
end

fixed_dofs = 1; % Node numbers of fixed DOFs
free_dofs = setxor(1:n_nodes,fixed_dofs); % Node numbers of free DOFs

%% Define Connectivity of Global Free DOFs
conn = zeros(n_blocks, 2); % Initialize element to node connectivity
for i = 1:n_blocks
    conn(i, 1) = i; % Local node 1
    conn(i, 2) = i+1; % Local node 2
end

%% Define Shape Functions
syms n 
N1n = 1/2*(1-n); N2n = 1/2*(1+n); % Linear shape functions in local coordinates
dndx = 2/dx; % Relationship between dx and dn in the form of dn/dx
dN1dn = diff(N1n); dN2dn = diff(N2n); % Derivative of shape functions wrt n
dN1dx = dN1dn*dndx; dN2dx = dN2dn*dndx; % Derivative of shape functions wrt x

%% Derive Parts of 'Stiffness' Matrix using Method of Weighted Residuals
k11eA = int(N1n^2, [-1 1]); k12eA = int(N1n*N2n, [-1 1]);
k21eA = int(N2n*N1n, [-1 1]); k22eA = int(N2n^2, [-1 1]);
kA = [k11eA, k12eA; k21eA, k22eA];

k11eC = double(int(dN1dx.^2, [-1 1])); k12eC = double(int(dN1dx.*dN2dx, [-1 1]));
k21eC = double(int(dN2dx.*dN1dx, [-1 1])); k22eC = double(int(dN2dx.^2, [-1 1]));
kC = [k11eC, k12eC; k21eC, k22eC];

%% Global 'Stiffness' Matrix Assembly (Apply Theta Differencing)
A = zeros(n_nodes, n_nodes); % Initialize Global 'Stiffness' Matrix for the next time step
B = zeros(n_nodes, n_nodes); % Initialize Global 'Stiffness' Matrix for the current time step
theta = 1/2; % Central difference method
beta = double(kA\(D*kC)); % [beta] in the hand calculation
for i = 1:n_blocks
    enodes = conn(i, :); % Element to node connectivity for local to global mapping
    ket1 = (1/dt*[2/3, 1/3; 1/3, 2/3] - (1-theta)*beta*[2/3, 1/3; 1/3, 2/3]); % Local 'Stiffness' Matrix for the current time step
    ket2 = (1/dt*[2/3, 1/3; 1/3, 2/3] + theta*beta*[2/3, 1/3; 1/3, 2/3]); % Local 'Stiffness' Matrix for the next time step
    A(enodes, enodes) = A(enodes, enodes) + ket2; % Assembly of the global 'stiffness' matrix for the next time step
    B(enodes, enodes) = B(enodes, enodes) + ket1; % Assembly of the global 'stiffness' matrix for the current time step
end

%% Global 'Stiffness' Matrix Partitioning
A_E = A(fixed_dofs, fixed_dofs);
A_EF = A(fixed_dofs, free_dofs);
A_FE = A(free_dofs, fixed_dofs);
A_F = A(free_dofs, free_dofs);

F_E = F(fixed_dofs); % Boundary Conditions at current time step

for j = 1:n_time
    %% Solving for Option Pricing for all Stock Price Increments and Time Steps
    if j >= 2
        F_F = F(free_dofs); % Free DOF Option pricing at current time step
        RS = B*F;
        F(2:end) = A_F\(RS(2:end) - A_FE*F_E);
    end
end

%% Solve for Exact Phi and Plot vs the Final Time Step Numerical Solution
% Exact Solutiion in Symbolic Form
syms x
C = phi_1*(1 - erf(x/(2*(D*t_total)^(1/2))));

figure(1)
fplot(x, C, [0, 0.1], 'b', 'LineWidth', 2)     % Exact Solution Plot
hold on
plot(x_coord, F, '--', 'LineWidth', 2)   % Approximate Solution Plot
hold off
title('Plot of Chloride Concentration Diffusion into Concrete')
xlabel('Depth of Concrete (m)')
ylabel('Concentration of Chloride by Weight of Concrete (%)')
legend('Exact Solution', 'FE Approximation')