clc, clear, close all
%% Given Data
K = 100; % Strike Price
Stock_Price = 150; % Current Stock Price
sigma = 60/100; % Implied Volatility
r = 0.02; % Assume 2 Percent. Close to US 10 Year Treasury Yield

T = 150/365.25; % Years Until Expiration
dtao = 1/365.25; % 1 Day Time Steps

L = 2*Stock_Price; % Maximum Stock Price
n_blocks = 50; % Number of Elements
dS = L/n_blocks; % Price Step

n_nodes = n_blocks + 1; % Linear Approximation
fixed_dofs = [1, n_nodes]; % Node numbers of fixed DOFs
free_dofs = setxor(1:n_nodes,fixed_dofs); % Node numbers of free DOFs

S = (0:dS:L)'; % Stock Price Vector from $0 to Maximum Stock Price
tao = (0:dtao:T)'; % Tao Vector, where tao = T - t
n_time = length(tao); % Number of Time Steps

V_1 = 0; % Option Price at Stock Price = $0 - Boundary Condition ($)
V_n = L - K*exp(-r*tao); % Option Price at Maximum Stock Price - Boundary Condition ($)
V = max(S-K, 0); % Free Global Option Price Vector at tao = 0 - Initial Condition ($)

V_total = zeros(n_nodes, n_time); % Initialize Option Price Storage Matrix for All Time Steps
V_total(:, 1) = V; % Initial Condition at tao = 0
V_total(end, :) = V_n; % Boundary Condition at Manually Set Maximum Stock Price

%% Define Connectivity of Global Free DOFs
conn = zeros(n_blocks, 2); % Initialize element to node connectivity
for i = 1:n_blocks
    conn(i, 1) = i; % Local node 1
    conn(i, 2) = i+1; % Local node 2
end

%% Define Shape Functions
syms n 
N1n = 1/2*(1-n); N2n = 1/2*(1+n); % Linear shape functions in local coordinates
dndS = 2/dS; % Relationship between dS and dn in the form of dn/dS
dN1dn = diff(N1n); dN2dn = diff(N2n); % Derivative of shape functions wrt n
dN1dS = dN1dn*dndS; dN2dS = dN2dn*dndS; % Derivative of shape functions wrt S

%% Derive Parts of 'Stiffness' Matrix using Method of Weighted Residuals
k11eB1 = int((1-n)*dN1dS*N1n, [-1 1]); k12eB1 = int((1-n)*dN1dS*N2n, [-1 1]);
k21eB1 = int((1-n)*dN2dS*N1n, [-1 1]); k22eB1 = int((1-n)*dN2dS*N2n, [-1 1]);
kB1 = [k11eB1, k12eB1; k21eB1, k22eB1];

k11eB2 = int((1+n)*dN1dS*N1n, [-1 1]); k12eB2 = int((1+n)*dN1dS*N2n, [-1 1]);
k21eB2 = int((1+n)*dN2dS*N1n, [-1 1]); k22eB2 = int((1+n)*dN2dS*N2n, [-1 1]);
kB2 = [k11eB2, k12eB2; k21eB2, k22eB2];

k11eD = int(N1n^2, [-1 1]); k12eD = int(N1n*N2n, [-1 1]);
k21eD = int(N2n*N1n, [-1 1]); k22eD = int(N2n^2, [-1 1]);
kD = [k11eD, k12eD; k21eD, k22eD];

k11eE1 = int((1-n)^2*dN1dS^2, [-1 1]); k12eE1 = int((1-n)^2*dN1dS*dN2dS, [-1 1]);
k21eE1 = int((1-n)^2*dN2dS*dN1dS, [-1 1]); k22eE1 = int((1-n)^2*dN2dS^2, [-1 1]);
kE1 = [k11eE1, k12eE1; k21eE1, k22eE1];

k11eE2 = int((1-n)*(1+n)*dN1dS^2, [-1 1]); k12eE2 = int((1-n)*(1+n)*dN1dS*dN2dS, [-1 1]);
k21eE2 = int((1-n)*(1+n)*dN2dS*dN1dS, [-1 1]); k22eE2 = int((1-n)*(1+n)*dN2dS^2, [-1 1]);
kE2 = [k11eE2, k12eE2; k21eE2, k22eE2];

k11eE3 = int((1+n)^2*dN1dS^2, [-1 1]); k12eE3 = int((1+n)^2*dN1dS*dN2dS, [-1 1]);
k21eE3 = int((1+n)^2*dN2dS*dN1dS, [-1 1]); k22eE3 = int((1+n)^2*dN2dS^2, [-1 1]);
kE3 = [k11eE3, k12eE3; k21eE3, k22eE3];

%% Matrix Initializations
A = zeros(n_nodes, n_nodes); % Free Global 'Stiffness' Matrix
B = zeros(n_nodes, 1); % Forcing Vector

%% Matrix Assembly
theta = 1/2; % Central difference method
for i = 1:n_blocks
    %% Global 'Stiffness' Matrix Assembly (Apply Theta Differencing)
    S1 = S(i); S2 = S(i+1);
    enodes = conn(i, :); % Element to node connectivity for local to global mapping
    m = kD\(2/dS*((sigma^2-r)*(1/2*S1*kB1 + 1/2*S2*kB2) + 1/8*...
        sigma^2*(S1^2*kE1 + 2*S1*S2*kE2 + S2^2*kE3) + r*kD)); % [m] in the hand calculation
    ke = (1/dtao + theta*m); % Local 'Stiffness' Matrix
    f = (1/dtao - (1-theta)*m); % Local Forcing Vector

    A(enodes, enodes) = A(enodes, enodes) + ke; % Assembly of the global 'stiffness' matrix
    B(enodes) = B(enodes) + f*V(enodes); % Assembly of the global forcing vector
end

% Filter only free DOFs as the fixed DOFs are known BCs
A_E = A(free_dofs, free_dofs);
V_E = V(free_dofs);
B_E = B(free_dofs);

for t = 2:n_time
    V_E = A_E\B_E;
    V_total(2:end-1, t) = V_E;
end
V_total;
%% Post Processing
% Filter V_total to Only Include More Realistic Stock Price Movements
% Range from -20% to +20% Current Stock Price
lower = Stock_Price*0.8;
upper = Stock_Price*1.2;
lowerbound_index = find(S == max(S(S <= lower)));
upperbound_index = find(S == min(S(S >= upper)));
V_filtered = V_total(lowerbound_index:upperbound_index, end:-1:1);
S_filtered = S(lowerbound_index:upperbound_index);

% Find the FDM Approximated Option Price at Current Stock Price and tao = T
current_index = find(S == max(S(S <= Stock_Price)));
% Linear Interpolation to Find Option Price In-Between Nodes
Option_Price = (V_total(current_index+1,end)-V_total(current_index,end))...
    /dS*(Stock_Price-S(current_index)) + V_total(current_index,end);
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