clc, clear, close all
%% Given Data
K = 25; % Strike Price
Stock_Price = 39.19; % Current Stock Price
sigma = 127.23/100; % Implied Volatility
r = 0.02; % Assume 2 Percent. Close to US 10 Year Treasury Yield

T = 20/365.25; % Years Until Expiration
dtao = 1/365.25; % 1 Day Time Steps

tao = (0:dtao:T)'; % Tao Vector, where tao = T - t
n_time = length(tao); % Number of Time Nodes

n_blocks = 30; % Number of Elements
n_nodes = n_blocks + 1; % Linear Approximation
fixed_dofs = [1, n_nodes]; % Node numbers of fixed DOFs
free_dofs = setxor(1:n_nodes,fixed_dofs); % Node numbers of free DOFs

L = log(2*Stock_Price) + (r - 1/2*sigma^2)*tao; % Upper limit on space dimension
dx = L/n_blocks; % Space Step
x = zeros(n_nodes, n_time);
for i = 1:n_time
    x(:, i) = (0:dx(i):L(i))'; % Space Vector, where x is a function of stock price and tao
end

F_1 = 0; % Option Price at $0 stock price - Boundary Condition
F_n = 2*Stock_Price - K*exp(-r*tao); % Option Price at max stock price - Boundary Condition
F = max(exp(x(:, 1))-K, 0); % Option Price at tao = 0 - Initial Condition

F_total = zeros(n_nodes, n_time); % Initialize Option Price Storage Matrix for All Time Steps
F_total(:, 1) = F; % Initial Condition at tao = 0
F_total(end, :) = F_n; % Boundary Condition at Manually Set Maximum Stock Price

%% Define Connectivity of Global Free DOFs
conn = zeros(n_blocks, 2); % Initialize element to node connectivity
for i = 1:n_blocks
    conn(i, 1) = i; % Local node 1
    conn(i, 2) = i+1; % Local node 2
end

%% Define Shape Functions
for j = 1:length(dx)
    syms n 
    N1n = 1/2*(1-n); N2n = 1/2*(1+n); % Linear shape functions in local coordinates
    dndS = 2/dx(j); % Relationship between dS and dn in the form of dn/dS
    dN1dn = diff(N1n); dN2dn = diff(N2n); % Derivative of shape functions wrt n
    dN1dS = dN1dn*dndS; dN2dS = dN2dn*dndS; % Derivative of shape functions wrt S

    %% Derive Parts of 'Stiffness' Matrix using Method of Weighted Residuals
    k11eA = int(N1n^2, [-1 1]); k12eA = int(N1n*N2n, [-1 1]);
    k21eA = int(N2n*N1n, [-1 1]); k22eA = int(N2n^2, [-1 1]);
    kA = [k11eA, k12eA; k21eA, k22eA];

    k11eC = int(dN1dS^2, [-1 1]); k12eC = int(dN1dS*dN2dS, [-1 1]);
    k21eC = int(dN2dS*dN1dS, [-1 1]); k22eC = int(dN2dS^2, [-1 1]);
    kC = [k11eC, k12eC; k21eC, k22eC];

    %% Global 'Stiffness' Matrix Assembly (Apply Theta Differencing)
    A = zeros(n_nodes, n_nodes); % Initialize Global 'Stiffness' Matrix for the next time step
    B = zeros(n_nodes, n_nodes); % Initialize Global 'Stiffness' Matrix for the current time step
    theta = 1/2; % Central difference method
    for i = 1:n_blocks
        S1 = x(i); S2 = x(i+1); % Local node stock price coordinates
        enodes = conn(i, :); % Element to node connectivity for local to global mapping
        beta = kA\(1/2*sigma^2*kC); % [beta] in the hand calculation
        ket1 = (1/dtao - 3*sigma^2/dx(j)*(1-theta)*beta); % Local 'Stiffness' Matrix for the current time step
        ket2 = (1/dtao + 3*sigma^2/dx(j)*theta*beta); % Local 'Stiffness' Matrix for the next time step
        A(enodes, enodes) = A(enodes, enodes) + ket2; % Assembly of the global 'stiffness' matrix
        B(enodes, enodes) = B(enodes, enodes) + ket1; % Assembly of the global forcing vector
    end

    %% Global 'Stiffness' Matrix Partitioning
    A_E = A(fixed_dofs, fixed_dofs);
    A_EF = A(fixed_dofs, free_dofs);
    A_FE = A(free_dofs, fixed_dofs);
    A_F = A(free_dofs, free_dofs);

    B_E = B(fixed_dofs, fixed_dofs);
    B_EF = B(fixed_dofs, free_dofs);
    B_FE = B(free_dofs, fixed_dofs);
    B_F = B(free_dofs, free_dofs);

    %% Solving for Option Pricing for all Stock Price Increments and Time Steps
    if j >= 2
        V_E = F(fixed_dofs); % Boundary Conditions at current time step
        V_F = F(free_dofs); % Free DOF Option pricing at current time step
        V_E2 = F_total(fixed_dofs, j); % Boundary Conditions at next time step
        V_F2 = A_F\(B_FE*V_E + B_F*V_F - A_FE*V_E2); % Free DOF Option pricing at next time step
        F_total(2:end-1, j) = V_F2;
        F = F_total(:, j);
    end
end
F_total;

%% Post Processing
% Filter V_total to Only Include More Realistic Stock Price Movements
% Range from -20% to +20% Current Stock Price
S = zeros(n_nodes, n_time);
for i = 1:n_time
    S(:, i) = exp(x(:, i) - (r-1/2*sigma^2)*tao(i));
end
S_0 = S(:, end);

V_total = zeros(n_nodes, n_time);
for i = 1:n_time
    V_total(:, i) = F_total(:, i)*exp(-r*tao(i));
end
V_total

lower = Stock_Price*0.8;
upper = Stock_Price*1.2;
lowerbound_index = find(S_0 == max(S_0(S_0 <= lower)));
upperbound_index = find(S_0 == min(S_0(S_0 >= upper)));
V_filtered = V_total(lowerbound_index:upperbound_index, end:-1:1);
S_filtered = S(lowerbound_index:upperbound_index, :);

% Find the FDM Approximated Option Price at Current Stock Price and tao = T
current_index = find(S_0 == max(S_0(S_0 <= Stock_Price)));
% Linear Interpolation to Find Option Price In-Between Nodes
dS_current = S_0(current_index+1)-S_0(current_index);
Option_Price = (V_total(current_index+1,end)-V_total(current_index,end))...
    /dS_current*(Stock_Price-S_0(current_index)) + V_total(current_index,end);
V_gainloss = (V_filtered - Option_Price)/Option_Price*100; % Percent Gain/Loss

%% Plotting
figure(1)
tao_days = tao*365.25; % Change time scale to days for graphing
surf(tao_days, S, V_total)
xlabel('Time Until Expiration (Days)')
ylabel('Stock Price ($)')
zlabel('Option Price ($)')
title('Black-Scholes Model - Call Option Pricing')

%% Print Results
fprintf('The current option price is $%.2f', Option_Price)